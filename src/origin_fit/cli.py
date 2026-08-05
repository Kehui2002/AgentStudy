from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Sequence, TextIO

from mini_agent import DeepSeekModelProvider, MiniAgentError

from .agent_cli import run_agent_cli
from .datasets import ImportSelection, import_dataset, inspect_dataset
from .errors import OriginFitError
from .execution import accept_fit_result
from .remote import HttpWorkerTransport, RemoteOriginExecutor
from .specifications import (
    approve_fit_specification,
    inspect_persisted_object,
    propose_fit_specification,
)
from .storage import LocalStore


def _mapping(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise OriginFitError("invalid_argument", f"{option} requires NAME=VALUE.")
        name, mapped = value.split("=", 1)
        if not name or not mapped or name in result:
            raise OriginFitError("invalid_argument", f"Invalid {option} mapping '{value}'.")
        result[name] = mapped
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="origin-fit")
    parser.add_argument("--state-dir", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    import_parser = commands.add_parser("import")
    import_parser.add_argument("csv_path", type=Path)
    import_parser.add_argument("--x", required=True)
    import_parser.add_argument("--y", action="append", required=True)
    import_parser.add_argument("--uncertainty", action="append", default=[])
    import_parser.add_argument("--unit", action="append", default=[])

    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("object_id")

    propose_parser = commands.add_parser("propose")
    propose_parser.add_argument("dataset_snapshot_id")
    propose_parser.add_argument("--experiment-id", required=True)
    propose_parser.add_argument("--fit-min", required=True, type=float)
    propose_parser.add_argument("--fit-max", required=True, type=float)
    propose_parser.add_argument(
        "--weighting", choices=("none", "instrument"), required=True
    )
    propose_parser.add_argument(
        "--initialization", choices=("origin-auto", "explicit"), required=True
    )
    propose_parser.add_argument("--initial-values", type=Path)
    propose_parser.add_argument("--graph-profile", required=True)

    approve_parser = commands.add_parser("approve")
    approve_parser.add_argument("fit_specification_id")
    accept_parser = commands.add_parser("accept")
    accept_parser.add_argument("fit_result_id")

    agent_parser = commands.add_parser("agent")
    agent_parser.add_argument("--worker-url", required=True)
    agent_parser.add_argument("--worker-certificate", required=True, type=Path)
    agent_parser.add_argument("--deepseek-model", default=None)
    return parser


def _graph_profile(value: str) -> tuple[str, str]:
    if "@" not in value:
        raise OriginFitError(
            "invalid_argument", "--graph-profile requires PROFILE_ID@VERSION."
        )
    profile_id, version = value.rsplit("@", 1)
    if not profile_id or not version:
        raise OriginFitError(
            "invalid_argument", "--graph-profile requires PROFILE_ID@VERSION."
        )
    return profile_id, version


def _initial_values(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OriginFitError(
            "invalid_fit_specification", "Initial values must be a readable UTF-8 JSON object."
        ) from error
    if not isinstance(value, dict):
        raise OriginFitError(
            "invalid_fit_specification", "Initial values must be a JSON object."
        )
    return value


async def _run_configured_agent(
    store: LocalStore,
    arguments: argparse.Namespace,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    worker_credential = os.environ.get("ORIGIN_WORKER_TOKEN")
    if not api_key or not worker_credential:
        raise OriginFitError(
            "missing_configuration", "Required credentials are not configured."
        )
    model_options = (
        {"model_name": arguments.deepseek_model}
        if arguments.deepseek_model is not None
        else {}
    )
    model = DeepSeekModelProvider(api_key, **model_options)
    try:
        transport = HttpWorkerTransport.with_pinned_certificate(
            arguments.worker_url,
            token=worker_credential,
            pinned_certificate=arguments.worker_certificate,
        )
    except ValueError as error:
        raise OriginFitError(
            "invalid_worker_configuration", "Origin Worker configuration is invalid."
        ) from error
    try:
        await run_agent_cli(
            store,
            model,
            RemoteOriginExecutor(transport),
            stdin=stdin,
            stdout=stdout,
        )
    finally:
        await transport.aclose()


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    input_stream = stdin or sys.stdin
    output = stdout or sys.stdout
    try:
        arguments = _parser().parse_args(argv)
        store = LocalStore(arguments.state_dir)
        result: dict
        if arguments.command == "agent":
            asyncio.run(
                _run_configured_agent(
                    store,
                    arguments,
                    stdin=input_stream,
                    stdout=output,
                )
            )
            return 0
        if arguments.command == "import":
            result = import_dataset(
                store,
                arguments.csv_path,
                ImportSelection(
                    x=arguments.x,
                    ys=tuple(arguments.y),
                    uncertainties=_mapping(arguments.uncertainty, "--uncertainty"),
                    units=_mapping(arguments.unit, "--unit"),
                ),
            )
        elif arguments.command == "inspect":
            persisted_object = inspect_persisted_object(store, arguments.object_id)
            result = (
                persisted_object
                if persisted_object is not None
                else inspect_dataset(store, arguments.object_id)
            )
        elif arguments.command == "propose":
            profile_id, profile_version = _graph_profile(arguments.graph_profile)
            result = propose_fit_specification(
                store,
                arguments.dataset_snapshot_id,
                experiment_id=arguments.experiment_id,
                fit_minimum=arguments.fit_min,
                fit_maximum=arguments.fit_max,
                weighting=arguments.weighting,
                initialization=arguments.initialization,
                graph_profile_id=profile_id,
                graph_profile_version=profile_version,
                initial_values=_initial_values(arguments.initial_values),
            )
        elif arguments.command == "approve":
            result = approve_fit_specification(
                store,
                arguments.fit_specification_id,
            )
        else:
            result = accept_fit_result(store, arguments.fit_result_id)
        print(json.dumps(result, sort_keys=True), file=output)
        return 0
    except OriginFitError as error:
        print(
            json.dumps({"error": error.code, "message": error.message}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except MiniAgentError:
        print(
            json.dumps(
                {
                    "error": "agent_run_failed",
                    "message": "Agent Run failed; check controlled local diagnostics.",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {
                    "error": "internal_error",
                    "message": "The command failed without exposing internal diagnostics.",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
