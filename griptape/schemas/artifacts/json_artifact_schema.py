from marshmallow import post_load, fields
from griptape.schemas import BaseArtifactSchema


class JsonArtifactSchema(BaseArtifactSchema):
    value = fields.Dict()

    @post_load
    def make_obj(self, data, **kwargs):
        from griptape.artifacts import JsonArtifact

        return JsonArtifact(**data)
