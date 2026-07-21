"""与模型服务商无关的模型接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..messages import ModelMessage, ModelResponse


@dataclass(frozen=True, slots=True)
class ModelRequestParameters:
    """Agent 在当前模型调用中要求启用的能力。"""

    allow_text_output: bool = True


class Model(ABC):
    """Agent 运行时与模型服务商之间的抽象边界。"""

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """返回一个经过标准化的模型响应。"""
        raise NotImplementedError
