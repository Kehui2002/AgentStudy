"""DeepSeek Chat Completions API 的 Model Provider 实现。"""

from __future__ import annotations

from typing import Any

import httpx

from ..exceptions import AgentRunError, UnexpectedModelBehavior, UserError
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UserPromptPart,
)
from .base import Model, ModelRequestParameters

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-v4-flash"


class DeepSeekModelProvider(Model):
    """通过 DeepSeek 的 OpenAI 兼容接口生成文本。

    ``http_client`` 是可选的依赖注入点，主要用于测试或由调用方统一管理连接池。
    未传入时，每次请求会创建并关闭一个独立的异步客户端。
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise UserError("DeepSeek api_key must not be empty")
        if not model_name:
            raise UserError("DeepSeek model_name must not be empty")
        if not base_url:
            raise UserError("DeepSeek base_url must not be empty")
        if timeout <= 0:
            raise UserError("DeepSeek timeout must be greater than 0")

        self._api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_client = http_client

    async def request(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """发送一次非流式文本请求，并返回标准化响应。"""
        if not parameters.allow_text_output:
            raise UserError(
                "DeepSeekModelProvider currently supports text output only"
            )

        payload = {
            "model": self.model_name,
            "messages": self._map_messages(messages),
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        if parameters.tool_definitions:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.parameters_json_schema,
                    },
                }
                for definition in parameters.tool_definitions
            ]
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise AgentRunError("DeepSeek request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise AgentRunError(f"DeepSeek API returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise AgentRunError(f"DeepSeek request failed: {exc}") from exc

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise UnexpectedModelBehavior("DeepSeek returned invalid JSON") from exc

        return self._map_response(data)

    @staticmethod
    def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """把框架内部消息历史转换成 DeepSeek 消息。"""
        mapped: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        mapped.append({"role": "user", "content": part.content})
                    elif isinstance(part, ToolResultPart):
                        mapped.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": part.content,
                            }
                        )
                    else:
                        raise UnexpectedModelBehavior(
                            "DeepSeekModelProvider cannot serialize request part "
                            f"{type(part).__name__}"
                        )
            elif isinstance(message, ModelResponse):
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.text or None,
                }
                tool_calls = [
                    part for part in message.parts if isinstance(part, ToolCallPart)
                ]
                if tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.arguments_json,
                            },
                        }
                        for part in tool_calls
                    ]
                mapped.append(assistant)
            else:
                raise UnexpectedModelBehavior(
                    "DeepSeekModelProvider cannot serialize message "
                    f"{type(message).__name__}"
                )

        return mapped

    @staticmethod
    def _map_response(data: Any) -> ModelResponse:
        """校验 DeepSeek 响应的最小结构并提取文本。"""
        if not isinstance(data, dict):
            raise UnexpectedModelBehavior("DeepSeek response must be a JSON object")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise UnexpectedModelBehavior("DeepSeek response did not contain choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise UnexpectedModelBehavior("DeepSeek response contained an invalid choice")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise UnexpectedModelBehavior("DeepSeek choice did not contain a message")

        parts: list[TextPart | ToolCallPart] = []
        content = message.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise UnexpectedModelBehavior(
                    "DeepSeek message contained invalid text content"
                )
            if content:
                parts.append(TextPart(content))

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise UnexpectedModelBehavior(
                    "DeepSeek message contained invalid tool calls"
                )
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    raise UnexpectedModelBehavior(
                        "DeepSeek message contained an invalid tool call"
                    )
                function = tool_call.get("function")
                if (
                    not isinstance(tool_call.get("id"), str)
                    or tool_call.get("type") != "function"
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise UnexpectedModelBehavior(
                        "DeepSeek message contained an invalid function tool call"
                    )
                parts.append(
                    ToolCallPart(
                        tool_call_id=tool_call["id"],
                        tool_name=function["name"],
                        arguments_json=function["arguments"],
                    )
                )

        if content is None and tool_calls is None:
            raise UnexpectedModelBehavior(
                "DeepSeek message did not contain text content or tool calls"
            )

        return ModelResponse(parts=parts)
