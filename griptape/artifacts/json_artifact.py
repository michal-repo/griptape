from __future__ import annotations
import json
from attr import define, field
from griptape.artifacts import TextArtifact, BaseArtifact


@define(frozen=True)
class JsonArtifact(TextArtifact):
    value: dict = field(converter=BaseArtifact.value_to_dict)

    def __add__(self, other: JsonArtifact) -> JsonArtifact:
        return JsonArtifact({**self.value, **other.value})

    def __str__(self):
        return self.to_text()

    def __bool__(self) -> bool:
        return len(self) > 0

    def to_text(self) -> str:
        return json.dumps(self.value)

    def to_dict(self) -> dict:
        from griptape.schemas import JsonArtifactSchema

        return dict(JsonArtifactSchema().dump(self))
