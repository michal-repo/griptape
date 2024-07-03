from __future__ import annotations
from typing import TYPE_CHECKING, Any
import json
from attrs import define, field
from griptape.artifacts import TextArtifact

if TYPE_CHECKING:
    from griptape.artifacts import BaseArtifact


@define
class JsonArtifact(TextArtifact):
    value: Any = field(converter=json.dumps, metadata={"serializable": True})  # pyright: ignore reportRedeclaration

    @property
    def value(self) -> Any:
        return json.loads(self.value)

    def to_text(self) -> str:
        return json.dumps(self.value)

    def __add__(self, other: BaseArtifact) -> JsonArtifact:
        new = json.loads(other.to_text())
        return JsonArtifact({**self.value, **new})

    def __eq__(self, value: object) -> bool:
        if isinstance(value, str):
            return self.to_text() == value
        return self.value == value
