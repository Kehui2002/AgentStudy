"""Provider-independent model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..messages import ModelMessage, ModelResponse


@dataclass(frozen=True, slots=True)
class ModelRequestParameters:
    """Capabilities requested by the agent for the current model call."""

    allow_text_output: bool = True


class Model(ABC):
    """Abstract boundary between the agent runtime and a model provider."""

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Return one normalized model response."""
        raise NotImplementedError
