from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence, TextIO

from .datasets import ImportSelection, import_dataset, inspect_dataset
from .errors import OriginFitError
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


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    output = stdout or sys.stdout
    try:
        arguments = _parser().parse_args(argv)
        store = LocalStore(arguments.state_dir)
        result: dict
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
        else:
            result = approve_fit_specification(
                store,
                arguments.fit_specification_id,
            )
        print(json.dumps(result, sort_keys=True), file=output)
        return 0
    except OriginFitError as error:
        print(
            json.dumps({"error": error.code, "message": error.message}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
