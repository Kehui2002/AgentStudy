"""输出解析与校验。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..exceptions import UnexpectedModelBehavior
from ..messages import ModelResponse

OutputT = TypeVar("OutputT")


class OutputSchema(ABC, Generic[OutputT]):
    """将标准化的模型响应转换为 Agent 的公开输出。"""

    @abstractmethod
    def validate(self, response: ModelResponse) -> OutputT:
        """校验并返回最终输出。"""
        raise NotImplementedError


class TextOutputSchema(OutputSchema[str]):
    """接收包含文本片段且不为空的响应。"""

    def validate(self, response: ModelResponse) -> str:
        if not response.parts:
            raise UnexpectedModelBehavior("model returned an empty response")

        text = response.text
        if not text:
            raise UnexpectedModelBehavior("model returned text parts without content")
        return text
