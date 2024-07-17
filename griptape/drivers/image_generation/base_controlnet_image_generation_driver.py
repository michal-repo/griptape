from abc import ABC

from griptape.artifacts import ImageArtifact


class BaseControlNetImageGenerationDriver(ABC):
    def try_control_net_image_generation(
        self, prompts: list[str], negative_prompts: list[str] | None = None
    ) -> ImageArtifact: ...
