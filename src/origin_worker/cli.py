from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from origin_fit.execution import DeterministicFakeOriginAdapter

from .api import create_app
from .service import OriginWorker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="origin-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--state-dir", required=True, type=Path)
    serve.add_argument("--host", required=True)
    serve.add_argument("--port", type=int, default=8443)
    serve.add_argument("--certfile", required=True, type=Path)
    serve.add_argument("--keyfile", required=True, type=Path)
    serve.add_argument("--fake-origin", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.host in {"0.0.0.0", "::"}:
        raise SystemExit("Origin Worker must bind to a specific host-only address.")
    if not arguments.certfile.is_file() or not arguments.keyfile.is_file():
        raise SystemExit("Origin Worker TLS certificate and key must exist.")
    token = os.environ.get("ORIGIN_WORKER_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("ORIGIN_WORKER_TOKEN must contain at least 32 characters.")
    if not arguments.fake_origin:
        raise SystemExit(
            "This build only includes the Fake Origin Adapter; pass --fake-origin."
        )

    import uvicorn

    worker = OriginWorker(arguments.state_dir, DeterministicFakeOriginAdapter())
    worker.health()
    app = create_app(worker, bearer_token=token)
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        ssl_certfile=str(arguments.certfile),
        ssl_keyfile=str(arguments.keyfile),
    )
    return 0
