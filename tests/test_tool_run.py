from __future__ import annotations

import asyncio
import json
import unittest
from enum import Enum

from pydantic import BaseModel, ConfigDict

from mini_agent import (
    Agent,
    Model,
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolError,
    ToolExecutionError,
    ToolRetryLimitExceeded,
    ToolResultPart,
    UserError,
    UserPromptPart,
)


class SequenceModel(Model):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.request_parameters: list[ModelRequestParameters] = []

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.request_parameters.append(parameters)
        return next(self._responses)


class Priority(str, Enum):
    HIGH = "high"
    LOW = "low"


class SearchWindow(BaseModel):
    start: int
    end: int | None


class SearchRequest(BaseModel):
    window: SearchWindow
    tags: list[str]
    priority: Priority
    weights: dict[str, float] | None


class SearchResult(BaseModel):
    selected_tags: list[str]
    priority: Priority
    window_end: int | None


class PermissiveSearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    selected_tags: list[str]


class ToolRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_rejects_non_finite_tool_arguments(self) -> None:
        def measure(value: float) -> float:
            """Measure a finite value."""
            return value

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-nan",
                            tool_name="measure",
                            arguments_json='{"value":NaN}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("I will use a finite value.")]),
            ]
        )

        result = await Agent(model, tools=[measure]).run("Measure the value")

        tool_result = result.all_messages()[2].parts[0]
        assert isinstance(tool_result, ToolResultPart)
        self.assertTrue(tool_result.is_error)
        self.assertEqual(
            json.loads(tool_result.content),
            {
                "error": "invalid_tool_arguments",
                "message": "Tool arguments failed validation.",
                "details": [
                    {
                        "path": ["value"],
                        "message": "Input must be a finite JSON number",
                    }
                ],
            },
        )

    async def test_agent_rejects_non_finite_tool_results(self) -> None:
        def measure(value: float) -> float:
            """Return a non-finite measurement."""
            return float("inf")

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-infinity",
                            tool_name="measure",
                            arguments_json='{"value":1.0}',
                        )
                    ]
                )
            ]
        )

        with self.assertRaisesRegex(ToolExecutionError, "Function Tool 'measure' failed"):
            await Agent(model, tools=[measure]).run("Measure the value")

    async def test_agent_rejects_extra_fields_on_a_return_model_instance(self) -> None:
        def search_catalog(query: str) -> PermissiveSearchResult:
            """Return a permissive model that violates the strict tool contract."""
            return PermissiveSearchResult.model_validate(
                {"selected_tags": [query], "secret": "must-not-be-returned"}
            )

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-extra-result",
                            tool_name="search_catalog",
                            arguments_json='{"query":"keyboard"}',
                        )
                    ]
                )
            ]
        )

        with self.assertRaisesRegex(
            ToolExecutionError, "Function Tool 'search_catalog' failed"
        ):
            await Agent(model, tools=[search_catalog]).run("Search products")

    async def test_agent_serializes_structured_results_with_stable_key_order(
        self,
    ) -> None:
        def summarize(query: str) -> dict[str, int]:
            """Return keys in a deliberately unstable-looking order."""
            return {"z_result": 2, "a_result": 1}

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-stable-json",
                            tool_name="summarize",
                            arguments_json='{"query":"keyboard"}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("Done.")]),
            ]
        )

        result = await Agent(model, tools=[summarize]).run("Summarize")

        tool_result = result.all_messages()[2].parts[0]
        assert isinstance(tool_result, ToolResultPart)
        self.assertEqual(tool_result.content, '{"a_result": 1, "z_result": 2}')

    async def test_agent_returns_a_recoverable_tool_error_and_continues(self) -> None:
        def submit_fit(recipe_id: str) -> dict[str, str]:
            """Submit an approved fit recipe."""
            raise ToolError(
                code="approval_required",
                message="The fit recipe requires approval.",
                details={"recipe_id": recipe_id},
            )

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-recoverable",
                            tool_name="submit_fit",
                            arguments_json='{"recipe_id":"recipe-7"}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("Please approve recipe-7 first.")]),
            ]
        )

        result = await Agent(model, tools=[submit_fit]).run("Run recipe-7")

        self.assertEqual(result.output, "Please approve recipe-7 first.")
        tool_result = result.all_messages()[2].parts[0]
        assert isinstance(tool_result, ToolResultPart)
        self.assertEqual(
            json.loads(tool_result.content),
            {
                "error": "approval_required",
                "message": "The fit recipe requires approval.",
                "details": {"recipe_id": "recipe-7"},
            },
        )

    async def test_agent_sanitizes_a_tool_error_with_non_json_details(self) -> None:
        def submit_fit(recipe_id: str) -> dict[str, str]:
            """Submit an approved fit recipe."""
            raise ToolError(
                code="remote_state",
                message="The remote state is unavailable.",
                details={"progress": float("nan")},
            )

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-invalid-error",
                            tool_name="submit_fit",
                            arguments_json='{"recipe_id":"recipe-7"}',
                        )
                    ]
                )
            ]
        )

        with self.assertRaisesRegex(
            ToolExecutionError, "Function Tool 'submit_fit' failed"
        ):
            await Agent(model, tools=[submit_fit]).run("Run recipe-7")

    async def test_agent_awaits_an_async_function_tool_without_blocking_the_loop(
        self,
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def lookup_inventory(product: str) -> dict[str, int | str]:
            """Look up inventory asynchronously."""
            started.set()
            await release.wait()
            return {"product": product, "stock": 7}

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-async",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":"keyboard"}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("There are 7 in stock.")]),
            ]
        )

        run_task = asyncio.create_task(
            Agent(model, tools=[lookup_inventory]).run("Check stock")
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        self.assertFalse(run_task.done())

        release.set()
        result = await asyncio.wait_for(run_task, timeout=1)

        tool_result = result.all_messages()[2].parts[0]
        assert isinstance(tool_result, ToolResultPart)
        self.assertEqual(
            json.loads(tool_result.content),
            {"product": "keyboard", "stock": 7},
        )

    async def test_agent_returns_nested_extra_fields_to_the_model_for_correction(
        self,
    ) -> None:
        def search_catalog(request: SearchRequest) -> SearchResult:
            """Search the catalog with structured criteria."""
            return SearchResult(
                selected_tags=request.tags,
                priority=request.priority,
                window_end=request.window.end,
            )

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-extra",
                            tool_name="search_catalog",
                            arguments_json=(
                                '{"request":{"window":{"start":1,"end":10,'
                                '"secret":"not-allowed"},"tags":[],"priority":"low",'
                                '"weights":null}}'
                            ),
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("I corrected the request.")]),
            ]
        )

        result = await Agent(model, tools=[search_catalog]).run("Search products")

        tool_result = result.all_messages()[2].parts[0]
        assert isinstance(tool_result, ToolResultPart)
        self.assertTrue(tool_result.is_error)
        self.assertEqual(
            json.loads(tool_result.content),
            {
                "error": "invalid_tool_arguments",
                "message": "Tool arguments failed validation.",
                "details": [
                    {
                        "path": ["request", "window", "secret"],
                        "message": "Extra inputs are not permitted",
                    }
                ],
            },
        )

    async def test_agent_runs_a_function_tool_with_structured_values(self) -> None:
        received: list[SearchRequest] = []

        def search_catalog(request: SearchRequest) -> SearchResult:
            """Search the catalog with structured criteria."""
            received.append(request)
            return SearchResult(
                selected_tags=request.tags,
                priority=request.priority,
                window_end=request.window.end,
            )

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-structured",
                            tool_name="search_catalog",
                            arguments_json=(
                                '{"request":{"window":{"start":1,"end":10},'
                                '"tags":["keyboards","wireless"],"priority":"high",'
                                '"weights":{"quality":0.75}}}'
                            ),
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("Found matching products.")]),
            ]
        )

        result = await Agent(model, tools=[search_catalog]).run("Search products")

        self.assertIsInstance(received[0], SearchRequest)
        self.assertIsInstance(received[0].window, SearchWindow)
        parameters_schema = model.request_parameters[0].tool_definitions[
            0
        ].parameters_json_schema
        self.assertFalse(parameters_schema["additionalProperties"])
        self.assertFalse(
            parameters_schema["$defs"]["SearchRequest"]["additionalProperties"]
        )
        self.assertFalse(
            parameters_schema["$defs"]["SearchWindow"]["additionalProperties"]
        )
        tool_result = result.all_messages()[2].parts[0]
        assert isinstance(tool_result, ToolResultPart)
        self.assertEqual(
            json.loads(tool_result.content),
            {
                "selected_tags": ["keyboards", "wireless"],
                "priority": "high",
                "window_end": 10,
            },
        )

    async def test_agent_returns_a_final_answer_after_a_function_tool(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            return {
                "product": product,
                "stock": 7,
                "warehouse": "Shanghai",
            }

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-1",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":"mechanical-keyboard"}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("上海仓库有 7 件库存。")]),
            ]
        )

        result = await Agent(model, tools=[lookup_inventory]).run("查询机械键盘库存")

        self.assertEqual(result.output, "上海仓库有 7 件库存。")
        self.assertEqual(
            result.all_messages(),
            [
                ModelRequest(parts=[UserPromptPart("查询机械键盘库存")]),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-1",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":"mechanical-keyboard"}',
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolResultPart(
                            tool_call_id="call-1",
                            content=(
                                '{"product": "mechanical-keyboard", "stock": 7, '
                                '"warehouse": "Shanghai"}'
                            ),
                            is_error=False,
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("上海仓库有 7 件库存。")]),
            ],
        )

    async def test_agent_reports_function_tool_usage(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            return {"product": product, "stock": 7}

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-1",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":"keyboard"}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("还有 7 件。")]),
            ]
        )

        result = await Agent(model, tools=[lookup_inventory]).run("查询库存")

        self.assertEqual(
            (
                result.usage.requests,
                result.usage.tool_calls,
                result.usage.tool_executions,
                result.usage.tool_retries,
            ),
            (2, 1, 1, 0),
        )

    async def test_agent_returns_invalid_arguments_to_the_model_for_correction(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            return {"product": product, "stock": 7}

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-invalid",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":123}',
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-corrected",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":"keyboard"}',
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("还有 7 件。")]),
            ]
        )

        result = await Agent(model, tools=[lookup_inventory]).run("查询库存")

        error_result = result.all_messages()[2].parts[0]
        self.assertIsInstance(error_result, ToolResultPart)
        assert isinstance(error_result, ToolResultPart)
        self.assertTrue(error_result.is_error)
        self.assertEqual(
            json.loads(error_result.content),
            {
                "error": "invalid_tool_arguments",
                "message": "Tool arguments failed validation.",
                "details": [
                    {
                        "path": ["product"],
                        "message": "Input should be a valid string",
                    }
                ],
            },
        )
        self.assertEqual(
            (
                result.usage.requests,
                result.usage.tool_calls,
                result.usage.tool_executions,
                result.usage.tool_retries,
            ),
            (3, 2, 1, 1),
        )

    async def test_agent_stops_after_the_function_tool_retry_limit(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            return {"product": product, "stock": 7}

        invalid_call = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_call_id="call-invalid",
                    tool_name="lookup_inventory",
                    arguments_json='{"product":123}',
                )
            ]
        )
        model = SequenceModel([invalid_call, invalid_call])

        with self.assertRaisesRegex(
            ToolRetryLimitExceeded,
            "max_tool_retries=1",
        ):
            await Agent(
                model,
                tools=[lookup_inventory],
                max_tool_retries=1,
            ).run("查询库存")

    async def test_agent_stops_when_function_tool_execution_fails(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            raise RuntimeError(
                "database credentials at /private/inventory.db must stay private"
            )

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-1",
                            tool_name="lookup_inventory",
                            arguments_json='{"product":"keyboard"}',
                        )
                    ]
                )
            ]
        )

        with self.assertRaisesRegex(
            ToolExecutionError,
            "Function Tool 'lookup_inventory' failed",
        ) as raised:
            await Agent(model, tools=[lookup_inventory]).run("查询库存")

        self.assertNotIn("credentials", str(raised.exception))
        self.assertNotIn("/private/inventory.db", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    def test_agent_rejects_a_function_tool_without_a_description(self) -> None:
        def lookup_inventory(product: str) -> dict:
            return {"product": product, "stock": 7}

        with self.assertRaisesRegex(
            UserError,
            "Function Tool 'lookup_inventory' must have a docstring",
        ):
            Agent(SequenceModel([]), tools=[lookup_inventory])

    def test_agent_rejects_unsupported_function_tool_signatures(self) -> None:
        def missing_parameter_annotation(product) -> dict:
            """缺少参数注解。"""
            return {"product": product}

        def missing_return_annotation(product: str):
            """缺少返回注解。"""
            return {"product": product}

        def variadic_parameter(*products: str) -> dict:
            """使用可变参数。"""
            return {"products": products}

        def positional_only_parameter(product: str, /) -> dict:
            """使用位置专用参数。"""
            return {"product": product}

        def invalid_default(unit: str = 1) -> dict:  # type: ignore[assignment]
            """默认值与参数类型不匹配。"""
            return {"unit": unit}

        for function in (
            missing_parameter_annotation,
            missing_return_annotation,
            variadic_parameter,
            positional_only_parameter,
            invalid_default,
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaises(UserError):
                    Agent(SequenceModel([]), tools=[function])

    def test_agent_rejects_duplicate_function_tool_names(self) -> None:
        def make_inventory_tool(stock: int):
            def lookup_inventory(product: str) -> dict:
                """查询产品的本地库存信息。"""
                return {"product": product, "stock": stock}

            return lookup_inventory

        with self.assertRaisesRegex(
            UserError,
            "Function Tool name 'lookup_inventory' is already registered",
        ):
            Agent(
                SequenceModel([]),
                tools=[make_inventory_tool(7), make_inventory_tool(8)],
            )

    async def test_agent_returns_an_unknown_function_tool_to_the_model(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            return {"product": product, "stock": 7}

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-unknown",
                            tool_name="invented_tool",
                            arguments_json="{}",
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("无法调用该工具。")]),
            ]
        )

        result = await Agent(model, tools=[lookup_inventory]).run("查询库存")

        error_result = result.all_messages()[2].parts[0]
        assert isinstance(error_result, ToolResultPart)
        self.assertEqual(
            json.loads(error_result.content),
            {
                "error": "unknown_tool",
                "message": "Function Tool 'invented_tool' is not registered.",
                "details": [],
            },
        )

    async def test_agent_returns_invalid_json_to_the_model(self) -> None:
        def lookup_inventory(product: str) -> dict:
            """查询产品的本地库存信息。"""
            return {"product": product, "stock": 7}

        model = SequenceModel(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_call_id="call-invalid-json",
                            tool_name="lookup_inventory",
                            arguments_json="{not-json",
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart("工具参数无法解析。")]),
            ]
        )

        result = await Agent(model, tools=[lookup_inventory]).run("查询库存")

        error_result = result.all_messages()[2].parts[0]
        assert isinstance(error_result, ToolResultPart)
        self.assertEqual(
            json.loads(error_result.content),
            {
                "error": "invalid_json",
                "message": "Tool arguments must be valid JSON.",
                "details": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
