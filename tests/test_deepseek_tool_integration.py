from __future__ import annotations

import os
import unittest

from mini_agent import Agent, DeepSeekModelProvider


def lookup_inventory(product: str) -> dict:
    """查询产品的本地库存信息。"""
    return {
        "product": product,
        "stock": 7,
        "warehouse": "Shanghai",
    }


@unittest.skipUnless(
    os.getenv("DEEPSEEK_API_KEY")
    and os.getenv("RUN_DEEPSEEK_INTEGRATION") == "1",
    "set DEEPSEEK_API_KEY and RUN_DEEPSEEK_INTEGRATION=1 to run",
)
class DeepSeekToolIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_agent_run_uses_a_function_tool(self) -> None:
        api_key = os.environ["DEEPSEEK_API_KEY"]
        agent = Agent(
            DeepSeekModelProvider(api_key=api_key),
            tools=[lookup_inventory],
        )

        result = await agent.run(
            "请务必调用 lookup_inventory 查询 mechanical-keyboard，"
            "然后告诉我库存地点和数量。"
        )

        self.assertTrue(result.output)
        self.assertGreaterEqual(result.usage.requests, 2)
        self.assertGreaterEqual(result.usage.tool_calls, 1)
        self.assertGreaterEqual(result.usage.tool_executions, 1)


if __name__ == "__main__":
    unittest.main()
