"""The public Agent API.

The public class intentionally delegates execution to ``AgentLoop``. Keeping the
API and runtime separate makes it possible to replace the explicit loop with a
node graph later without changing user code.
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
    """A minimal text-output agent.

    Args:
        model: Model implementation used to generate a response.
        max_steps: Safety limit for the number of model request steps in one run.
    """

    def __init__(self, model: Model, *, max_steps: int = 10) -> None:
        if max_steps < 1:
            raise UserError("max_steps must be at least 1")

        self.model = model
        self.max_steps = max_steps
        self._output_schema = TextOutputSchema()
        self._loop = AgentLoop()

    async def run(self, prompt: str) -> AgentResult[str]:
        """Run the agent asynchronously and return its validated text result."""
        ctx = RunContext(
            model=self.model,
            prompt=prompt,
            output_schema=self._output_schema,
            max_steps=self.max_steps,
        )
        return await self._loop.run(ctx)

    def run_sync(self, prompt: str) -> AgentResult[str]:
        """Synchronous convenience wrapper around :meth:`run`.

        Like most ``run_sync`` APIs, this must not be called from an already
        running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(prompt))

        raise UserError("run_sync() cannot be called while an event loop is running; use await agent.run() instead")
