"""Deterministic model used by examples and tests."""

from __future__ import annotations

from copy import deepcopy

from ..messages import ModelMessage, ModelResponse, TextPart
from .base import Model, ModelRequestParameters


class FakeModel(Model):
    """Always return the configured response and record every request."""

    def __init__(self, response: str | ModelResponse = "This is a fake response.") -> None:
        self._response = ModelResponse(parts=[TextPart(response)]) if isinstance(response, str) else response
        self.requests: list[list[ModelMessage]] = []
        self.request_parameters: list[ModelRequestParameters] = []

    async def request(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Record inputs and return a defensive copy of the configured response."""
        self.requests.append(deepcopy(messages))
        self.request_parameters.append(parameters)
        return deepcopy(self._response)
