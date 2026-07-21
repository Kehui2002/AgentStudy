"""第一学习阶段使用的显式 Agent 循环。"""

from __future__ import annotations

from ..context import RunContext
from ..exceptions import StepLimitExceeded, UnexpectedModelBehavior, UserError
from ..messages import ModelRequest, ModelResponse, UserPromptPart
from ..models import ModelRequestParameters
from ..result import AgentResult
from .state import AgentState


class AgentLoop:
    """持续驱动模型请求，直到输出模式生成最终结果。

    第一阶段只支持文本输出，因此第一个有效响应就会结束循环。后续阶段会加入
    工具调用和重试分支，通过 ``continue`` 进入下一次模型请求。
    """

    async def run(self, ctx: RunContext) -> AgentResult[str]:
        """使用传入的上下文执行一次相互独立的运行。"""
        if not ctx.prompt:
            raise UserError("prompt must not be empty in learning stage one")

        state = AgentState()
        state.append(ModelRequest(parts=[UserPromptPart(ctx.prompt)]))

        while state.run_step < ctx.max_steps:
            response = await ctx.model.request(
                state.all_messages(),
                ModelRequestParameters(allow_text_output=True),
            )
            if not isinstance(response, ModelResponse):
                raise UnexpectedModelBehavior(
                    f"Model.request() must return ModelResponse, got {type(response).__name__}"
                )

            state.usage.requests += 1
            state.run_step += 1
            state.append(response)

            output = ctx.output_schema.validate(response)
            return AgentResult(
                output=output,
                usage=state.usage.copy(),
                _messages=state.all_messages(),
            )

        raise StepLimitExceeded(f"agent exceeded max_steps={ctx.max_steps}")
