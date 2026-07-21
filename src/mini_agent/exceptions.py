"""迷你 Agent 框架抛出的异常。"""


class MiniAgentError(Exception):
    """本项目所有异常的基类。"""


class UserError(MiniAgentError):
    """配置无效，或公开 API 的使用方式无效。"""


class AgentRunError(MiniAgentError):
    """Agent 运行时无法正常完成任务。"""


class UnexpectedModelBehavior(AgentRunError):
    """模型返回了当前运行时无法处理的响应。"""


class StepLimitExceeded(AgentRunError):
    """本次运行超过了配置的模型调用步数限制。"""
