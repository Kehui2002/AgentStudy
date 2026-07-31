"""使用真实 DeepSeek API 运行一次 Function Tool 闭环。"""

from __future__ import annotations

import os

from mini_agent import Agent, DeepSeekModelProvider, ToolCallPart, ToolResultPart


def lookup_inventory(product: str) -> dict:
    """查询产品的本地库存信息。"""
    return {
        "product": product,
        "stock": 7,
        "warehouse": "Shanghai",
    }


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set; load the project .env file first"
        )

    agent = Agent(
        DeepSeekModelProvider(api_key=api_key),
        tools=[lookup_inventory],
    )
    result = agent.run_sync(
        "请务必调用 lookup_inventory 查询 mechanical-keyboard，"
        "然后告诉我库存地点和数量。"
    )

    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"Tool Call: {part.tool_name}({part.arguments_json})")
            elif isinstance(part, ToolResultPart):
                label = "Tool Error" if part.is_error else "Tool Result"
                print(f"{label}: {part.content}")

    print(f"Final Answer: {result.output}")
    print(
        "Usage: "
        f"requests={result.usage.requests}, "
        f"tool_calls={result.usage.tool_calls}, "
        f"tool_executions={result.usage.tool_executions}, "
        f"tool_retries={result.usage.tool_retries}"
    )


if __name__ == "__main__":
    main()
