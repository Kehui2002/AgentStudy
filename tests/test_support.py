"""Shared test support for Registered Origin Graph Template fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

from origin_worker.service import OriginWorker
from origin_worker.templates import GraphTemplateRegistry


TEMPLATE_NAME = "standard"
TEMPLATE_CONTENT = b"ORIGIN-GRAPH-TEMPLATE-CONTENT\n"
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE_CONTENT).hexdigest()
TEMPLATE_ID = "template:standard"
TEMPLATE_VERSION = 1


def template_selection() -> dict:
    return {
        "template_id": TEMPLATE_ID,
        "version": TEMPLATE_VERSION,
        "sha256": TEMPLATE_SHA256,
    }


def template_cli_arguments() -> tuple[str, ...]:
    return (
        "--template",
        f"{TEMPLATE_ID}@{TEMPLATE_VERSION}",
        "--template-sha256",
        TEMPLATE_SHA256,
    )


def register_standard_template(
    registry: GraphTemplateRegistry,
    *,
    content: bytes = TEMPLATE_CONTENT,
    name: str = TEMPLATE_NAME,
) -> dict:
    return registry.register(
        name=name,
        content=content,
        filename=f"{name}.otpu",
        graph_profile_id="expdec2-standard",
        graph_profile_version="1.0",
        originpro_min_version=10.2,
        originpro_max_version=10.3,
    )


def make_worker(
    state_dir: Path,
    adapter: object,
    **kwargs: object,
) -> OriginWorker:
    registry = GraphTemplateRegistry(state_dir)
    register_standard_template(registry)
    return OriginWorker(
        state_dir,
        adapter,  # type: ignore[arg-type]
        template_registry=registry,
        **kwargs,  # type: ignore[arg-type]
    )
