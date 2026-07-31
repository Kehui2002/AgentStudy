"""使用真实 DeepSeek API 运行最小文本 Agent。"""

import asyncio
import os

from mini_agent import Agent, DeepSeekModelProvider


async def main() -> None:
    model = DeepSeekModelProvider(api_key=os.environ["DEEPSEEK_API_KEY"])
    result = await Agent(model).run("用一句话解释什么是 Agent。")

    print(result.output)
    print(result.usage)


if __name__ == "__main__":
    asyncio.run(main())
