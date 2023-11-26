from abc import ABC
from attr import define, field
from griptape.mixins import ExponentialBackoffMixin


@define
class BaseImageDriver(ExponentialBackoffMixin, ABC):
    model: str = field(kw_only=True)
