"""Agent 运行过程中累计的用量计数。"""

from dataclasses import dataclass


@dataclass(slots=True)
class RunUsage:
    """一次 Agent Run 的模型请求与 Function Tool 用量。"""

    requests: int = 0
    tool_calls: int = 0
    tool_executions: int = 0
    tool_retries: int = 0

    def copy(self) -> "RunUsage":
        """为公开结果返回一份独立的状态快照。"""
        return RunUsage(
            requests=self.requests,
            tool_calls=self.tool_calls,
            tool_executions=self.tool_executions,
            tool_retries=self.tool_retries,
        )
