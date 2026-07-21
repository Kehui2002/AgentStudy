"""Output parsing and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..exceptions import UnexpectedModelBehavior
from ..messages import ModelResponse

OutputT = TypeVar("OutputT")


class OutputSchema(ABC, Generic[OutputT]):
    """Turn a normalized model response into the agent's public output."""

    @abstractmethod
    def validate(self, response: ModelResponse) -> OutputT:
        """Validate and return the final output."""
        raise NotImplementedError


class TextOutputSchema(OutputSchema[str]):
    """Accept a non-empty response containing text parts."""

    def validate(self, response: ModelResponse) -> str:
        if not response.parts:
            raise UnexpectedModelBehavior("model returned an empty response")

        text = response.text
        if not text:
            raise UnexpectedModelBehavior("model returned text parts without content")
        return text
