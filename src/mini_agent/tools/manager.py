"""Function Tool 的参数校验与执行。"""

from __future__ import annotations

import inspect
import json
import logging
import math
from collections.abc import Callable, Iterable
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from ..exceptions import ToolError, ToolExecutionError
from ..messages import ToolCallPart, ToolResultPart
from .definition import ToolDefinition
from .registry import ToolRegistry

_logger = logging.getLogger(__name__)


def _non_finite_number_details(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> list[dict[str, JsonValue]]:
    """Locate numbers that JSON cannot represent without extensions."""
    if isinstance(value, float) and not math.isfinite(value):
        return [
            {
                "path": list(path),
                "message": "Input must be a finite JSON number",
            }
        ]
    if isinstance(value, dict):
        return [
            detail
            for key, item in value.items()
            for detail in _non_finite_number_details(item, (*path, str(key)))
        ]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            detail
            for index, item in enumerate(value)
            for detail in _non_finite_number_details(item, (*path, index))
        ]
    return []


class ToolManager:
    """执行由 Model Provider 返回的 Tool Call。"""

    def __init__(self, tools: Iterable[Callable[..., Any]]) -> None:
        self._registry = ToolRegistry(tools)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._registry.definitions()

    @staticmethod
    def _error_result(
        call: ToolCallPart,
        *,
        code: str,
        message: str,
        details: JsonValue | None = None,
    ) -> ToolResultPart:
        return ToolResultPart(
            tool_call_id=call.tool_call_id,
            content=json.dumps(
                {
                    "error": code,
                    "message": message,
                    "details": details if details is not None else [],
                },
                ensure_ascii=False,
            ),
            is_error=True,
        )

    async def execute(self, call: ToolCallPart) -> ToolResultPart:
        execution_failed = False
        try:
            tool = self._registry.get(call.tool_name)
        except KeyError:
            return self._error_result(
                call,
                code="unknown_tool",
                message=f"Function Tool {call.tool_name!r} is not registered.",
            )

        try:
            arguments = tool.arguments_model.model_validate_json(
                call.arguments_json,
                strict=True,
                extra="forbid",
            )
        except ValidationError as exc:
            if any(error["type"] == "json_invalid" for error in exc.errors()):
                return self._error_result(
                    call,
                    code="invalid_json",
                    message="Tool arguments must be valid JSON.",
                )

            details: list[dict[str, JsonValue]] = [
                {
                    "path": list(error["loc"]),
                    "message": error["msg"],
                }
                for error in exc.errors()
            ]
            return self._error_result(
                call,
                code="invalid_tool_arguments",
                message="Tool arguments failed validation.",
                details=cast(JsonValue, details),
            )

        non_finite_arguments = _non_finite_number_details(
            arguments.model_dump(mode="python")
        )
        if non_finite_arguments:
            return self._error_result(
                call,
                code="invalid_tool_arguments",
                message="Tool arguments failed validation.",
                details=cast(JsonValue, non_finite_arguments),
            )

        try:
            validated_arguments = {
                name: getattr(arguments, name)
                for name in tool.arguments_model.model_fields
            }
            result = tool.function(**validated_arguments)
            if inspect.isawaitable(result):
                result = await result
            validated_result = tool.return_adapter.validate_python(
                result,
                strict=True,
                extra="forbid",
            )
            non_finite_result = _non_finite_number_details(
                tool.return_adapter.dump_python(validated_result, mode="python")
            )
            if non_finite_result:
                raise ValueError("Function Tool result contains a non-finite number")

            # Revalidate the serialized value so Pydantic cannot reuse a model
            # instance whose own config permits extra fields.
            validated_result = tool.return_adapter.validate_json(
                tool.return_adapter.dump_json(validated_result),
                strict=True,
                extra="forbid",
            )
            json_result = tool.return_adapter.dump_python(
                validated_result,
                mode="json",
            )
            content = (
                validated_result
                if isinstance(validated_result, str)
                else json.dumps(
                    json_result,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except ToolError as exc:
            return self._error_result(
                call,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        except Exception:
            _logger.debug(
                "Function Tool %r failed unexpectedly",
                call.tool_name,
                exc_info=True,
            )
            execution_failed = True

        if execution_failed:
            raise ToolExecutionError(f"Function Tool {call.tool_name!r} failed")

        return ToolResultPart(
            tool_call_id=call.tool_call_id,
            content=content,
            is_error=False,
        )
