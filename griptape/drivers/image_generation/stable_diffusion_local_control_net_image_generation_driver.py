import io

from attrs import define, field

from griptape.artifacts import ImageArtifact
from griptape.drivers.image_generation.base_control_net_image_generation_driver import (
    BaseControlNetImageGenerationDriver,
)
from griptape.drivers.image_generation.stable_diffusion_local_image_generation_driver import (
    StableDiffusionLocalImageGenerationDriver,
)

from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
import torch
from PIL import Image


@define
class StableDiffusionLocalControlNetImageGenerationDriver(
    StableDiffusionLocalImageGenerationDriver, BaseControlNetImageGenerationDriver
):
    control_net_model: str = field(kw_only=True)
    controlnet_conditioning_scale: float | None = field(default=None, kw_only=True, metadata={"serializable": True})

    def try_controlnet_image_generation(
        self, prompts: list[str], control_image: ImageArtifact, negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        control_image_input = Image.open(io.BytesIO(control_image.value))

        controlnet = SD3ControlNetModel.from_pretrained(self.control_net_model, torch_dtype=torch.float16)
        controlnet.to(self.device)

        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.model, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.to(self.device)

        output = pipe(
            ", ".join(prompts),
            control_image=control_image_input,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        ).images[0]

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        return ImageArtifact(value=buffer.getvalue(), format="png", height=self.height, width=self.width)
