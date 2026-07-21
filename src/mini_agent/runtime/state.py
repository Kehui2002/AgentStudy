"""由单次 Agent 运行独占的可变状态。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..messages import ModelMessage
from ..usage import RunUsage


@dataclass(slots=True)
class AgentState:
    """随着 Agent 循环推进而变化的状态。"""

    message_history: list[ModelMessage] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    run_step: int = 0

    def append(self, message: ModelMessage) -> None:
        """向对话历史追加一个请求或响应。"""
        self.message_history.append(message)

    def all_messages(self) -> list[ModelMessage]:
        """返回一份相互隔离的历史快照。"""
        return deepcopy(self.message_history)
