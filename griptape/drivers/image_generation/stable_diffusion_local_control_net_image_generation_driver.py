import io
import os
from typing import Any

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
    controlnet_model: str = field(kw_only=True)
    controlnet_conditioning_scale: float | None = field(default=None, kw_only=True, metadata={"serializable": True})

    def _load_pipeline(
        self, pipeline_class: type[StableDiffusion3ControlNetPipeline], controlnet: Any | None = None
    ) -> Any:
        if os.path.exists(self.model):
            return pipeline_class.from_single_file(self.model, controlnet=controlnet, torch_dtype=torch.float16)

        return pipeline_class.from_pretrained(self.model, controlnet=controlnet, torch_dtype=torch.float16)

    def _load_controlnet(self, controlnet_class: type[SD3ControlNetModel]) -> Any:
        if os.path.exists(self.controlnet_model):
            return SD3ControlNetModel.from_single_file(self.controlnet_model, torch_dtype=torch.float16)

        return SD3ControlNetModel.from_pretrained(self.controlnet_model, torch_dtype=torch.float16)

    def try_controlnet_image_generation(
        self, prompts: list[str], control_image: ImageArtifact, negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        control_image_input = Image.open(io.BytesIO(control_image.value))

        controlnet = self._load_controlnet(SD3ControlNetModel)
        controlnet.to(self.device)

        pipe = self._load_pipeline(StableDiffusion3ControlNetPipeline, controlnet=controlnet)
        pipe.to(self.device)

        output = pipe(
            ", ".join(prompts),
            control_image=control_image_input,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        ).images[0]

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        return ImageArtifact(value=buffer.getvalue(), format="png", height=self.height, width=self.width)
