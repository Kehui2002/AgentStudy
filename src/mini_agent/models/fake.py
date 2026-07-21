"""供示例和测试使用的确定性模型。"""

from __future__ import annotations

from copy import deepcopy

from ..messages import ModelMessage, ModelResponse, TextPart
from .base import Model, ModelRequestParameters


class FakeModel(Model):
    """始终返回预先配置的响应，并记录每一次请求。"""

    def __init__(self, response: str | ModelResponse = "This is a fake response.") -> None:
        self._response = ModelResponse(parts=[TextPart(response)]) if isinstance(response, str) else response
        self.requests: list[list[ModelMessage]] = []
        self.request_parameters: list[ModelRequestParameters] = []

    async def request(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """记录输入，并返回预设响应的防御性副本。"""
        self.requests.append(deepcopy(messages))
        self.request_parameters.append(parameters)
        return deepcopy(self._response)
