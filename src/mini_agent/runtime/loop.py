"""The explicit agent loop used in learning stage one."""

from __future__ import annotations

from ..context import RunContext
from ..exceptions import StepLimitExceeded, UnexpectedModelBehavior, UserError
from ..messages import ModelRequest, ModelResponse, UserPromptPart
from ..models import ModelRequestParameters
from ..result import AgentResult
from .state import AgentState


class AgentLoop:
    """Drive model requests until an output schema produces a final result.

    Stage one only supports text output, so a valid first response ends the
    loop. Later stages will add tool and retry branches that ``continue`` back
    to the next model request.
    """

    async def run(self, ctx: RunContext) -> AgentResult[str]:
        """Execute one independent run using the supplied context."""
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
