"""迷你 Agent 框架抛出的异常。"""

from __future__ import annotations

from pydantic import ConfigDict, JsonValue, TypeAdapter

_TOOL_ERROR_DETAILS_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(
    JsonValue,
    config=ConfigDict(allow_inf_nan=False),
)


class MiniAgentError(Exception):
    """本项目所有异常的基类。"""


class UserError(MiniAgentError):
    """配置无效，或公开 API 的使用方式无效。"""


class AgentRunError(MiniAgentError):
    """Agent 运行时无法正常完成任务。"""


class UnexpectedModelBehavior(AgentRunError):
    """模型返回了当前运行时无法处理的响应。"""


class StepLimitExceeded(AgentRunError):
    """本次运行超过了配置的模型调用步数限制。"""


class ToolRetryLimitExceeded(AgentRunError):
    """本次 Agent Run 超过了工具纠错次数限制。"""


class ToolError(MiniAgentError):
    """Function Tool 显式报告的可恢复、可安全返回模型的领域错误。"""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: JsonValue | None = None,
    ) -> None:
        if not isinstance(code, str) or not code:
            raise UserError("ToolError code must be a non-empty string")
        if not isinstance(message, str) or not message:
            raise UserError("ToolError message must be a non-empty string")

        super().__init__(message)
        self.code = code
        self.message = message
        self.details: JsonValue = _TOOL_ERROR_DETAILS_ADAPTER.validate_python(
            {} if details is None else details,
            strict=True,
        )


class ToolExecutionError(AgentRunError):
    """Function Tool 本身无法正常完成执行。"""
