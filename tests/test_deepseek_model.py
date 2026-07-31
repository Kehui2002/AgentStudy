from __future__ import annotations

import json
import unittest
from enum import Enum

import httpx
from pydantic import BaseModel

from mini_agent import (
    Agent,
    AgentRunError,
    DeepSeekModelProvider,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
    UnexpectedModelBehavior,
    UserPromptPart,
)
from mini_agent.models import ModelRequestParameters


class SortDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class SearchOptions(BaseModel):
    tags: list[str]
    direction: SortDirection
    limit: int | None
    selector: str | int


class DeepSeekModelProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_maps_generated_complex_function_tool_schema(self) -> None:
        captured_json: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_json.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "No search needed."}}]},
            )

        def search_catalog(options: SearchOptions) -> dict[str, int]:
            """Search a catalog with structured options."""
            return {"matches": len(options.tags)}

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DeepSeekModelProvider(api_key="test-key", http_client=client)
            await Agent(provider, tools=[search_catalog]).run("Find products")

        tools = captured_json["tools"]
        assert isinstance(tools, list)
        schema = tools[0]["function"]["parameters"]
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["SearchOptions"]["additionalProperties"])
        properties = schema["$defs"]["SearchOptions"]["properties"]
        self.assertEqual(properties["tags"]["items"], {"type": "string"})
        self.assertEqual(
            properties["direction"], {"$ref": "#/$defs/SortDirection"}
        )
        self.assertEqual(
            properties["limit"]["anyOf"],
            [{"type": "integer"}, {"type": "null"}],
        )
        self.assertEqual(
            properties["selector"]["anyOf"],
            [{"type": "string"}, {"type": "integer"}],
        )

    async def test_request_maps_function_tool_definitions(self) -> None:
        captured_json: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_json.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "最终回答"}}]},
            )

        definition = ToolDefinition(
            name="lookup_inventory",
            description="查询产品的本地库存信息。",
            parameters_json_schema={
                "type": "object",
                "properties": {"product": {"type": "string"}},
                "required": ["product"],
            },
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeekModelProvider(api_key="test-key", http_client=client)
            await model.request(
                [ModelRequest(parts=[UserPromptPart("查询库存")])],
                ModelRequestParameters(tool_definitions=(definition,)),
            )

        self.assertEqual(captured_json["thinking"], {"type": "disabled"})
        self.assertEqual(captured_json["tool_choice"], "auto")
        self.assertEqual(
            captured_json["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_inventory",
                        "description": "查询产品的本地库存信息。",
                        "parameters": {
                            "type": "object",
                            "properties": {"product": {"type": "string"}},
                            "required": ["product"],
                        },
                    },
                }
            ],
        )

    async def test_response_maps_text_and_function_tool_calls(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "我先查询库存。",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup_inventory",
                                            "arguments": (
                                                '{"product":"mechanical-keyboard"}'
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeekModelProvider(api_key="test-key", http_client=client)
            response = await model.request(
                [ModelRequest(parts=[UserPromptPart("查询库存")])],
                ModelRequestParameters(),
            )

        self.assertEqual(
            response,
            ModelResponse(
                parts=[
                    TextPart("我先查询库存。"),
                    ToolCallPart(
                        tool_call_id="call-1",
                        tool_name="lookup_inventory",
                        arguments_json='{"product":"mechanical-keyboard"}',
                    ),
                ]
            ),
        )

    async def test_message_history_maps_function_tool_calls_and_results(self) -> None:
        captured_json: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_json.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "还有 7 件。"}}]},
            )

        messages: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[UserPromptPart("查询库存")]),
            ModelResponse(
                parts=[
                    TextPart("我先查询。"),
                    ToolCallPart(
                        tool_call_id="call-1",
                        tool_name="lookup_inventory",
                        arguments_json='{"product":"keyboard"}',
                    ),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolResultPart(
                        tool_call_id="call-1",
                        content='{"product":"keyboard","stock":7}',
                        is_error=False,
                    )
                ]
            ),
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeekModelProvider(api_key="test-key", http_client=client)
            await model.request(messages, ModelRequestParameters())

        self.assertEqual(
            captured_json["messages"],
            [
                {"role": "user", "content": "查询库存"},
                {
                    "role": "assistant",
                    "content": "我先查询。",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup_inventory",
                                "arguments": '{"product":"keyboard"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": '{"product":"keyboard","stock":7}',
                },
            ],
        )

    async def test_agent_maps_request_and_response(self) -> None:
        captured_request: httpx.Request | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "来自 DeepSeek 的回答"}}
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeekModelProvider(api_key="test-key", http_client=client)
            result = await Agent(model).run("你好")

        self.assertEqual(result.output, "来自 DeepSeek 的回答")
        self.assertEqual(result.usage.requests, 1)
        self.assertIsNotNone(captured_request)
        assert captured_request is not None
        self.assertEqual(captured_request.url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured_request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(
            json.loads(captured_request.content),
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "你好"}],
                "thinking": {"type": "disabled"},
                "stream": False,
            },
        )

    async def test_message_history_maps_user_and_assistant_roles(self) -> None:
        captured_json: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_json.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "第二个回答"}}]},
            )

        messages: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[UserPromptPart("第一个问题")]),
            ModelResponse(parts=[TextPart("第一个回答")]),
            ModelRequest(parts=[UserPromptPart("第二个问题")]),
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeekModelProvider(api_key="test-key", http_client=client)
            response = await model.request(messages, ModelRequestParameters())

        self.assertEqual(response, ModelResponse(parts=[TextPart("第二个回答")]))
        self.assertEqual(
            captured_json["messages"],
            [
                {"role": "user", "content": "第一个问题"},
                {"role": "assistant", "content": "第一个回答"},
                {"role": "user", "content": "第二个问题"},
            ],
        )

    async def test_http_error_becomes_agent_run_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "invalid key"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(
                DeepSeekModelProvider(api_key="bad-key", http_client=client)
            )
            with self.assertRaisesRegex(AgentRunError, "HTTP 401"):
                await agent.run("你好")

    async def test_malformed_response_is_rejected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(
                DeepSeekModelProvider(api_key="test-key", http_client=client)
            )
            with self.assertRaisesRegex(UnexpectedModelBehavior, "did not contain choices"):
                await agent.run("你好")


if __name__ == "__main__":
    unittest.main()
