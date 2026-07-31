"""Function Tool 注册表。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..exceptions import UserError
from .definition import FunctionTool, ToolDefinition


class ToolRegistry:
    """按名称保存已注册的 Function Tool。"""

    def __init__(self, tools: Iterable[Callable[..., Any]]) -> None:
        self._tools: dict[str, FunctionTool] = {}
        for function in tools:
            tool = FunctionTool.from_callable(function)
            name = tool.definition.name
            if name in self._tools:
                raise UserError(
                    f"Function Tool name {name!r} is already registered"
                )
            self._tools[name] = tool

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def get(self, name: str) -> FunctionTool:
        return self._tools[name]
