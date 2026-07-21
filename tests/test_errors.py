from __future__ import annotations

import unittest

from mini_agent import Agent, FakeModel, ModelResponse, UnexpectedModelBehavior, UserError


class ErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_prompt_is_rejected_before_model_request(self) -> None:
        model = FakeModel("unused")

        with self.assertRaisesRegex(UserError, "prompt must not be empty"):
            await Agent(model).run("")

        self.assertEqual(model.requests, [])

    async def test_empty_model_response_is_rejected(self) -> None:
        agent = Agent(FakeModel(ModelResponse()))

        with self.assertRaisesRegex(UnexpectedModelBehavior, "empty response"):
            await agent.run("hello")

    def test_invalid_step_limit_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(UserError, "max_steps must be at least 1"):
            Agent(FakeModel(), max_steps=0)


if __name__ == "__main__":
    unittest.main()
