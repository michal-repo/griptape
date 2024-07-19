from abc import ABC

from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseImageGenerationDriver


class BaseControlNetImageGenerationDriver(BaseImageGenerationDriver, ABC):
    def run_controlnet_image_generation(
        self, prompts: list[str], control_image: ImageArtifact, negative_prompts: list[str] | None
    ) -> ImageArtifact:
        for attempt in self.retrying():
            with attempt:
                self.before_run(prompts, negative_prompts)
                result = self.try_controlnet_image_generation(prompts, control_image, negative_prompts)
                self.after_run()

                return result

        else:
            raise Exception("Failed to run ControlNet image generation")

    def try_controlnet_image_generation(
        self, prompts: list[str], control_image: ImageArtifact, negative_prompts: list[str] | None = None
    ) -> ImageArtifact: ...
