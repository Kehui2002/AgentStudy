"""Mutable state owned by a single agent run."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..messages import ModelMessage
from ..usage import RunUsage


@dataclass(slots=True)
class AgentState:
    """State that changes as the agent loop advances."""

    message_history: list[ModelMessage] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    run_step: int = 0

    def append(self, message: ModelMessage) -> None:
        """Append one request or response to the conversation history."""
        self.message_history.append(message)

    def all_messages(self) -> list[ModelMessage]:
        """Return an isolated history snapshot."""
        return deepcopy(self.message_history)
