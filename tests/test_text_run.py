from __future__ import annotations

import unittest

from mini_agent import Agent, FakeModel, ModelRequest, ModelResponse, TextPart, UserPromptPart


class TextRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_text_run_returns_output_and_usage(self) -> None:
        model = FakeModel("hello from the fake model")
        agent = Agent(model)

        result = await agent.run("hello")

        self.assertEqual(result.output, "hello from the fake model")
        self.assertEqual(result.usage.requests, 1)
        self.assertEqual(len(model.requests), 1)
        self.assertTrue(model.request_parameters[0].allow_text_output)

    async def test_message_history_records_request_then_response(self) -> None:
        model = FakeModel("answer")
        result = await Agent(model).run("question")

        self.assertEqual(
            result.all_messages(),
            [
                ModelRequest(parts=[UserPromptPart("question")]),
                ModelResponse(parts=[TextPart("answer")]),
            ],
        )

        # 模型只能看到生成本次响应之前已经存在的消息历史。
        self.assertEqual(
            model.requests,
            [[ModelRequest(parts=[UserPromptPart("question")])]],
        )

    def test_sync_wrapper_uses_the_same_runtime(self) -> None:
        result = Agent(FakeModel("sync answer")).run_sync("question")

        self.assertEqual(result.output, "sync answer")
        self.assertEqual(result.usage.requests, 1)


if __name__ == "__main__":
    unittest.main()
