from attrs import define, field

from griptape.artifacts import ImageArtifact
from griptape.drivers.image_generation.stable_diffusion_local_control_net_image_generation_driver import (
    StableDiffusionLocalControlNetImageGenerationDriver,
)
from griptape.loaders import ImageLoader
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from schema import Schema, Literal


@define
class ControlNetImageGenerationTool(BaseTool):
    driver: StableDiffusionLocalControlNetImageGenerationDriver = field(kw_only=True, metadata={"serializable": True})

    @activity(
        config={
            "description": "Used to generate images using Stable Diffusion 3 and ControlNet. Image generations require "
            "a prompt (positively weighted descriptions of the resulting image), a list"
            "negative prompt (negatively weighted descriptions of the resulting image), "
            "and a control image. For both positive and negative prompts, more detail is better. "
            "Both prompts are limited to 77 tokens in length.",
            "schema": Schema(
                {
                    Literal("prompt", description="A detailed description of the desired image and its qualities"): str,
                    Literal(
                        "negative_prompt",
                        description="A detailed description of undesirable qualities of " "the output image",
                    ): str,
                    Literal("control_image_path", description="Path to the control image"): str,
                }
            ),
        }
    )
    def generate_image(self, params: dict[str, any]) -> ImageArtifact:
        prompt = params["values"]["prompt"]
        negative_prompt = params["values"]["negative_prompt"]
        control_image_path = params["values"]["control_image_path"]

        with open(control_image_path, "rb") as f:
            control_image = ImageLoader().load(f.read())

        return self.driver.run_controlnet_image_generation([prompt], control_image, [negative_prompt])
