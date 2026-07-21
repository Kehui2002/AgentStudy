"""内部运行时可见的单次运行依赖。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Model
from .output import OutputSchema


@dataclass(frozen=True, slots=True)
class RunContext:
    """单次 Agent 运行所需的配置和依赖。

    可变的执行数据应放在 ``AgentState`` 中，而不是这里。
    """

    model: Model
    prompt: str
    output_schema: OutputSchema[str]
    max_steps: int
