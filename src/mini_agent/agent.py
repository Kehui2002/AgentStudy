"""Agent 的公开 API。

公开类有意将执行过程委托给 ``AgentLoop``。将 API 与运行时分离后，未来可以在
不修改用户代码的前提下，用节点图替换当前的显式循环。
"""

from __future__ import annotations

import asyncio

from .context import RunContext
from .exceptions import UserError
from .models import Model
from .output import TextOutputSchema
from .result import AgentResult
from .runtime import AgentLoop


class Agent:
    """一个只输出文本的最小 Agent。

    参数：
        model: 用于生成响应的模型实现。
        max_steps: 单次运行允许执行模型请求的最大步数，用作安全限制。
    """

    def __init__(self, model: Model, *, max_steps: int = 10) -> None:
        if max_steps < 1:
            raise UserError("max_steps must be at least 1")

        self.model = model
        self.max_steps = max_steps
        self._output_schema = TextOutputSchema()
        self._loop = AgentLoop()

    async def run(self, prompt: str) -> AgentResult[str]:
        """异步运行 Agent，并返回通过校验的文本结果。"""
        ctx = RunContext(
            model=self.model,
            prompt=prompt,
            output_schema=self._output_schema,
            max_steps=self.max_steps,
        )
        return await self._loop.run(ctx)

    def run_sync(self, prompt: str) -> AgentResult[str]:
        """:meth:`run` 的同步便捷封装。

        与大多数 ``run_sync`` API 一样，不能在已经运行的事件循环中调用它。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(prompt))

        raise UserError("run_sync() cannot be called while an event loop is running; use await agent.run() instead")
