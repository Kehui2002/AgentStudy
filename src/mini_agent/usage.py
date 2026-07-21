"""Agent 运行过程中累计的用量计数。"""

from dataclasses import dataclass


@dataclass(slots=True)
class RunUsage:
    """第一学习阶段使用的最小用量信息。"""

    requests: int = 0

    def copy(self) -> "RunUsage":
        """为公开结果返回一份独立的状态快照。"""
        return RunUsage(requests=self.requests)
