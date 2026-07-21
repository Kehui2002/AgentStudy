"""Per-run dependencies visible to the internal runtime."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Model
from .output import OutputSchema


@dataclass(frozen=True, slots=True)
class RunContext:
    """Configuration and dependencies for one agent run.

    Mutable execution data belongs in ``AgentState`` rather than here.
    """

    model: Model
    prompt: str
    output_schema: OutputSchema[str]
    max_steps: int
