"""Public result returned by an agent run."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .messages import ModelMessage
from .usage import RunUsage

OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class AgentResult(Generic[OutputT]):
    """The validated output plus observable state from one agent run."""

    output: OutputT
    usage: RunUsage
    _messages: list[ModelMessage] = field(repr=False)

    def all_messages(self) -> list[ModelMessage]:
        """Return a defensive copy of the complete message history."""
        return deepcopy(self._messages)
