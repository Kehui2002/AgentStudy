from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import sqlite3
from typing import Sequence

from origin_fit.execution import DeterministicFakeOriginAdapter

from .api import create_app
from .originpro_adapter import OriginProAdapter, OriginProAdapterError
from .service import OriginWorker, WorkerError
from .templates import GraphTemplateRegistry, TemplateError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="origin-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--state-dir", required=True, type=Path)
    serve.add_argument("--host", required=True)
    serve.add_argument("--host-only-network", required=True)
    serve.add_argument("--linux-guest-address", required=True)
    serve.add_argument("--port", type=int, default=8443)
    serve.add_argument("--certfile", required=True, type=Path)
    serve.add_argument("--keyfile", required=True, type=Path)
    serve.add_argument("--fake-origin", action="store_true")
    serve.add_argument("--origin-visible", action="store_true")
    template = commands.add_parser("template")
    template_commands = template.add_subparsers(dest="template_command", required=True)
    register = template_commands.add_parser("register")
    register.add_argument("--state-dir", required=True, type=Path)
    register.add_argument("--name", required=True)
    register.add_argument("--file", required=True, type=Path)
    register.add_argument("--graph-profile", required=True)
    register.add_argument("--originpro-min", required=True, type=float)
    register.add_argument("--originpro-max", required=True, type=float)
    for command_name in ("list", "show", "deactivate"):
        subcommand = template_commands.add_parser(command_name)
        subcommand.add_argument("--state-dir", required=True, type=Path)
        if command_name in ("show", "deactivate"):
            subcommand.add_argument("template_reference")
    return parser


def _template_reference(value: str) -> tuple[str, int]:
    if "@" not in value:
        raise SystemExit(
            "Template reference must be TEMPLATE_ID@VERSION, e.g. template:standard@1."
        )
    template_id, raw_version = value.rsplit("@", 1)
    try:
        version = int(raw_version)
    except ValueError as error:
        raise SystemExit(
            "Template reference version must be an integer."
        ) from error
    if version < 1:
        raise SystemExit("Template reference version must be at least 1.")
    return template_id, version


def _run_template_command(arguments: argparse.Namespace) -> int:
    registry = GraphTemplateRegistry(arguments.state_dir)
    try:
        if arguments.template_command == "register":
            profile_id, profile_version = arguments.graph_profile.rsplit("@", 1)
            if not profile_id or not profile_version:
                raise TemplateError(
                    "invalid_template_metadata",
                    "--graph-profile requires PROFILE_ID@VERSION.",
                )
            try:
                content = arguments.file.read_bytes()
            except OSError as error:
                raise TemplateError(
                    "template_unreadable",
                    "Registered Origin Graph Template file could not be read.",
                ) from error
            result = registry.register(
                name=arguments.name,
                content=content,
                filename=arguments.file.name,
                graph_profile_id=profile_id,
                graph_profile_version=profile_version,
                originpro_min_version=arguments.originpro_min,
                originpro_max_version=arguments.originpro_max,
            )
        elif arguments.template_command == "list":
            result = {"graph_templates": registry.list_templates()}
        else:
            template_id, version = _template_reference(arguments.template_reference)
            if arguments.template_command == "show":
                shown = registry.get(template_id, version)
                if shown is None:
                    raise TemplateError(
                        "template_not_found",
                        f"Registered Origin Graph Template '{template_id}@{version}' not found.",
                    )
                result = shown
            else:
                result = registry.deactivate(template_id, version)
    except TemplateError as error:
        raise SystemExit(f"{error.code}: {error.message}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "template":
        return _run_template_command(arguments)
    try:
        host = ipaddress.ip_address(arguments.host)
        host_only_network = ipaddress.ip_network(
            arguments.host_only_network, strict=False
        )
        linux_guest = ipaddress.ip_address(arguments.linux_guest_address)
    except ValueError as error:
        raise SystemExit("Origin Worker host-only address/network is invalid.") from error
    if (
        host not in host_only_network
        or not host.is_private
        or not host_only_network.is_private
        or host.is_unspecified
        or host.is_loopback
        or host.is_link_local
        or host.is_multicast
    ):
        raise SystemExit(
            "Origin Worker host must belong to the declared private host-only network."
        )
    if (
        linux_guest not in host_only_network
        or linux_guest == host
        or not linux_guest.is_private
        or linux_guest.is_unspecified
        or linux_guest.is_loopback
        or linux_guest.is_link_local
        or linux_guest.is_multicast
        or linux_guest == host_only_network.network_address
        or linux_guest == host_only_network.broadcast_address
    ):
        raise SystemExit(
            "Linux guest address must be a distinct private unicast address in the "
            "declared host-only network."
        )
    if not arguments.certfile.is_file() or not arguments.keyfile.is_file():
        raise SystemExit("Origin Worker TLS certificate and key must exist.")
    if arguments.certfile.stat().st_size == 0 or arguments.keyfile.stat().st_size == 0:
        raise SystemExit("Origin Worker TLS certificate and key must not be empty.")
    token = os.environ.get("ORIGIN_WORKER_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("ORIGIN_WORKER_TOKEN must contain at least 32 characters.")
    if arguments.fake_origin and arguments.origin_visible:
        raise SystemExit("--origin-visible cannot be combined with --fake-origin.")
    state_reference = str(arguments.state_dir)
    if (
        state_reference.startswith(("\\\\", "//"))
        or arguments.state_dir.is_symlink()
        or (arguments.state_dir.exists() and not arguments.state_dir.is_dir())
    ):
        raise SystemExit(
            "Origin Worker state directory must be a local, non-symlink directory."
        )

    adapter = None
    try:
        adapter = (
            DeterministicFakeOriginAdapter()
            if arguments.fake_origin
            else OriginProAdapter(visible=arguments.origin_visible)
        )
        preflight = getattr(adapter, "preflight", None)
        if callable(preflight):
            preflight()
        worker = OriginWorker(arguments.state_dir, adapter)
        worker.health()
    except (OSError, sqlite3.Error, OriginProAdapterError, WorkerError) as error:
        terminate = getattr(adapter, "terminate", None)
        if callable(terminate):
            terminate()
        raise SystemExit(
            "Origin Worker state directory, SQLite, or Adapter preflight failed."
        ) from error

    import uvicorn

    app = create_app(
        worker,
        bearer_token=token,
        allowed_client_hosts={str(linux_guest)},
    )
    try:
        uvicorn.run(
            app,
            host=arguments.host,
            port=arguments.port,
            ssl_certfile=str(arguments.certfile),
            ssl_keyfile=str(arguments.keyfile),
        )
    finally:
        terminate = getattr(adapter, "terminate", None)
        if callable(terminate):
            terminate()
    return 0
