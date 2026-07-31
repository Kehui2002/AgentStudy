"""模型抽象与测试用实现。"""

from .base import Model, ModelRequestParameters
from .deepseek import DeepSeekModelProvider
from .fake import FakeModel

__all__ = (
    "DeepSeekModelProvider",
    "FakeModel",
    "Model",
    "ModelRequestParameters",
)
