"""模型抽象与测试用实现。"""

from .base import Model, ModelRequestParameters
from .fake import FakeModel

__all__ = ("FakeModel", "Model", "ModelRequestParameters")
