"""驱动文本输出与 Function Tool 的显式 Agent 循环。"""

from __future__ import annotations

from ..context import RunContext
from ..exceptions import (
    StepLimitExceeded,
    ToolRetryLimitExceeded,
    UnexpectedModelBehavior,
    UserError,
)
from ..messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolResultPart,
    UserPromptPart,
)
from ..models import ModelRequestParameters
from ..result import AgentResult
from .state import AgentState


class AgentLoop:
    """持续驱动模型请求，直到输出模式生成最终结果。

    包含 Tool Call 的响应会先执行全部 Function Tool，并通过 ``continue`` 把
    Tool Result 送回模型；不包含 Tool Call 的有效文本响应会结束运行。
    """

    async def run(self, ctx: RunContext) -> AgentResult[str]:
        """使用传入的上下文执行一次相互独立的运行。"""
        if not ctx.prompt:
            raise UserError("prompt must not be empty")

        state = AgentState()
        state.append(ModelRequest(parts=[UserPromptPart(ctx.prompt)]))

        while state.run_step < ctx.max_steps:
            response = await ctx.model.request(
                state.all_messages(),
                ModelRequestParameters(
                    allow_text_output=True,
                    tool_definitions=ctx.tool_manager.definitions(),
                ),
            )
            if not isinstance(response, ModelResponse):
                raise UnexpectedModelBehavior(
                    f"Model.request() must return ModelResponse, got {type(response).__name__}"
                )

            state.usage.requests += 1
            state.run_step += 1
            state.append(response)

            tool_calls = [
                part for part in response.parts if isinstance(part, ToolCallPart)
            ]
            if tool_calls:
                state.usage.tool_calls += len(tool_calls)
                tool_results = [
                    await ctx.tool_manager.execute(call) for call in tool_calls
                ]
                state.usage.tool_executions += sum(
                    not result.is_error for result in tool_results
                )
                state.usage.tool_retries += sum(
                    result.is_error for result in tool_results
                )
                if state.usage.tool_retries > ctx.max_tool_retries:
                    raise ToolRetryLimitExceeded(
                        f"agent exceeded max_tool_retries={ctx.max_tool_retries}"
                    )
                request_parts: list[UserPromptPart | ToolResultPart] = list(
                    tool_results
                )
                state.append(ModelRequest(parts=request_parts))
                continue

            output = ctx.output_schema.validate(response)
            return AgentResult(
                output=output,
                usage=state.usage.copy(),
                _messages=state.all_messages(),
            )

        raise StepLimitExceeded(f"agent exceeded max_steps={ctx.max_steps}")
