"""Exceptions raised by the mini agent framework."""


class MiniAgentError(Exception):
    """Base exception for this project."""


class UserError(MiniAgentError):
    """Invalid configuration or invalid public API usage."""


class AgentRunError(MiniAgentError):
    """The agent runtime could not complete normally."""


class UnexpectedModelBehavior(AgentRunError):
    """The model returned a response the current runtime cannot handle."""


class StepLimitExceeded(AgentRunError):
    """The run exceeded its configured model-step limit."""
