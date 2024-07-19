from typing import Callable

from attrs import define, field

from griptape.artifacts import TextArtifact, ImageArtifact, ListArtifact
from griptape.tasks import BaseImageGenerationTask, BaseTask
from griptape.drivers.image_generation.stable_diffusion_local_control_net_image_generation_driver import (
    StableDiffusionLocalControlNetImageGenerationDriver,
)
from griptape.utils import J2


@define
class StableDiffusionControlNetImageGenerationTask(BaseImageGenerationTask):
    _driver: StableDiffusionLocalControlNetImageGenerationDriver = field(kw_only=True, metadata={"serializable": True})
    _input: (
        tuple[str | TextArtifact, TextArtifact, ImageArtifact] | Callable[[BaseTask], ListArtifact] | ListArtifact
    ) = field(default=None)

    @property
    def input(self) -> ListArtifact:
        if isinstance(self._input, ListArtifact):
            return self._input
        elif isinstance(self._input, tuple):
            if isinstance(self._input[0], TextArtifact):
                prompt_input_text = self._input[0]
            else:
                prompt_input_text = TextArtifact(J2().render_from_string(self._input[0], **self.full_context))

            if isinstance(self._input[1], TextArtifact):
                negative_prompt_input_text = self._input[1]
            else:
                negative_prompt_input_text = TextArtifact(J2().render_from_string(self._input[1], **self.full_context))

            return ListArtifact([prompt_input_text, negative_prompt_input_text, self._input[2]])
        elif isinstance(self._input, Callable):
            return self._input(self)
        else:
            raise ValueError("Input must be a tuple of (text, image) or a callable that returns such a tuple.")

    @input.setter
    def input(self, value: tuple[str | TextArtifact, ImageArtifact] | Callable[[BaseTask], ListArtifact]) -> None:
        self._input = value

    @property
    def driver(self) -> StableDiffusionLocalControlNetImageGenerationDriver:
        if self._driver is None:
            raise ValueError("Image Generation Engine is not set.")

        return self._driver

    @driver.setter
    def driver(self, value: StableDiffusionLocalControlNetImageGenerationDriver) -> None:
        self._driver = value

    def run(self) -> ImageArtifact:
        prompt_artifact = self.input[0]
        negative_prompt_artifact = self.input[1]
        control_image_artifact = self.input[2]

        if not isinstance(control_image_artifact, ImageArtifact):
            raise ValueError("Control image must be an ImageArtifact.")

        output_image_artifact = self.driver.run_controlnet_image_generation(
            prompts=[prompt_artifact.to_text()],
            control_image=control_image_artifact,
            negative_prompts=[negative_prompt_artifact.to_text()],
        )

        if self.output_dir or self.output_file:
            self._write_to_file(output_image_artifact)

        return output_image_artifact
