"""Agent 运行时使用的、与模型服务商无关的消息协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class UserPromptPart:
    """模型请求中包含的用户输入。"""

    content: str


@dataclass(frozen=True, slots=True)
class TextPart:
    """模型生成的文本。"""

    content: str


ModelRequestPart: TypeAlias = UserPromptPart
ModelResponsePart: TypeAlias = TextPart


@dataclass(slots=True)
class ModelRequest:
    """发送给模型的、与服务商无关的请求。"""

    parts: list[ModelRequestPart] = field(default_factory=list)


@dataclass(slots=True)
class ModelResponse:
    """模型返回的、与服务商无关的响应。"""

    parts: list[ModelResponsePart] = field(default_factory=list)

    @property
    def text(self) -> str:
        """按照响应顺序拼接所有文本片段。"""
        return "".join(part.content for part in self.parts)


ModelMessage: TypeAlias = ModelRequest | ModelResponse
