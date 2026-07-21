"""Agent 运行后返回的公开结果。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .messages import ModelMessage
from .usage import RunUsage

OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class AgentResult(Generic[OutputT]):
    """单次运行中通过校验的输出，以及可供观察的状态。"""

    output: OutputT
    usage: RunUsage
    _messages: list[ModelMessage] = field(repr=False)

    def all_messages(self) -> list[ModelMessage]:
        """返回完整消息历史的防御性副本。"""
        return deepcopy(self._messages)
