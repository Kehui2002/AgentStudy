"""Function Tool 的模型描述与 Python 可调用对象。"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

from ..exceptions import UserError

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _forbid_extra_model_fields(schema: Any) -> None:
    """Make nested Pydantic model objects as strict as runtime validation."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema.setdefault("additionalProperties", False)
        for value in schema.values():
            _forbid_extra_model_fields(value)
    elif isinstance(schema, list):
        for value in schema:
            _forbid_extra_model_fields(value)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """提供给 Model Provider 的 Function Tool 描述。"""

    name: str
    description: str
    parameters_json_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """注册后的 Function Tool 及其 Pydantic 校验器。"""

    definition: ToolDefinition
    function: Callable[..., Any]
    arguments_model: type[BaseModel]
    return_adapter: TypeAdapter[Any]

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> FunctionTool:
        if not _TOOL_NAME_PATTERN.fullmatch(function.__name__):
            raise UserError(
                f"Function Tool name {function.__name__!r} is not valid"
            )

        description = inspect.getdoc(function)
        if not description:
            raise UserError(
                f"Function Tool {function.__name__!r} must have a docstring"
            )

        signature = inspect.signature(function)
        try:
            type_hints = get_type_hints(function)
        except (NameError, TypeError) as exc:
            raise UserError(
                f"Function Tool {function.__name__!r} has invalid type annotations"
            ) from exc
        fields: dict[str, tuple[Any, Any]] = {}

        for name, parameter in signature.parameters.items():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise UserError(
                    f"Function Tool {function.__name__!r} parameter {name!r} "
                    "must accept keyword arguments"
                )
            if name not in type_hints:
                raise UserError(
                    f"Function Tool {function.__name__!r} parameter {name!r} "
                    "must have a type annotation"
                )

            annotation = type_hints[name]
            default = (
                ...
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            if default is not ...:
                try:
                    TypeAdapter(annotation).validate_python(default, strict=True)
                except Exception as exc:
                    raise UserError(
                        f"Function Tool {function.__name__!r} parameter {name!r} "
                        "has an invalid default value"
                    ) from exc
            fields[name] = (annotation, default)

        if "return" not in type_hints:
            raise UserError(
                f"Function Tool {function.__name__!r} must have a return annotation"
            )
        return_annotation = type_hints["return"]
        try:
            arguments_model = create_model(  # type: ignore[call-overload]
                f"{function.__name__}Arguments",
                __config__=ConfigDict(
                    extra="forbid", strict=True, validate_default=True
                ),
                **fields,
            )
            parameters_json_schema = arguments_model.model_json_schema()
            _forbid_extra_model_fields(parameters_json_schema)
            return_adapter = TypeAdapter(return_annotation)
            return_adapter.json_schema()
        except Exception as exc:
            raise UserError(
                f"Function Tool {function.__name__!r} must use "
                "JSON-schema-compatible annotations"
            ) from exc

        definition = ToolDefinition(
            name=function.__name__,
            description=description,
            parameters_json_schema=parameters_json_schema,
        )
        return cls(
            definition=definition,
            function=function,
            arguments_model=arguments_model,
            return_adapter=return_adapter,
        )
