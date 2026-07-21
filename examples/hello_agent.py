"""Run the first-stage text agent without calling an external API."""

import asyncio

from mini_agent import Agent, FakeModel


async def main() -> None:
    agent = Agent(FakeModel("你好，我已经走完最小 Agent 调用链。"))
    result = await agent.run("你好")

    print(result.output)
    print(result.all_messages())
    print(result.usage)


if __name__ == "__main__":
    asyncio.run(main())
