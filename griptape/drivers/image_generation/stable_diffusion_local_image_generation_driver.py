from typing import Optional
import io

from attrs import define, field

from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseImageGenerationDriver
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
import torch
from PIL import Image


@define
class StableDiffusionLocalImageGenerationDriver(BaseImageGenerationDriver):
    model: str = field(default="stabilityai/stable-diffusion-3-medium-diffusers", kw_only=True)
    device: str = field(default="cpu", kw_only=True)
    seed: int | None = field(default=None, kw_only=True, metadata={"serializable": True})
    guidance_scale: float | None = field(default=None, kw_only=True, metadata={"serializable": True})
    steps: int | None = field(default=None, kw_only=True, metadata={"serializable": True})
    width: int = field(default=1024, kw_only=True, metadata={"serializable": True})
    height: int = field(default=1024, kw_only=True, metadata={"serializable": True})
    strength: float | None = field(default=None, kw_only=True, metadata={"serializable": True})

    def try_text_to_image(self, prompts: list[str], negative_prompts: list[str] | None = None) -> ImageArtifact:
        if isinstance(prompts, str):
            prompt = prompts
        else:
            prompt = ", ".join(prompts)

        pipeline = StableDiffusion3Pipeline.from_pretrained(self.model, torch_dtype=torch.float16)
        pipeline.to(self.device)
        image = pipeline(
            prompt, width=self.width, height=self.height, **self._make_additional_params(negative_prompts)
        ).images[0]

        # save PIL image to buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return ImageArtifact(value=buffer.getvalue(), format="png", height=512, width=512)

    def try_image_variation(
        self, prompts: list[str], image: ImageArtifact, negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        prompt = ", ".join(prompts)

        input_image = Image.open(io.BytesIO(image.value))
        pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(self.model, torch_dtype=torch.float16)
        pipeline.to(self.device)
        image = pipeline(
            prompt,
            width=self.width,
            height=self.height,
            image=input_image,
            **self._make_additional_params(negative_prompts),
        )

        # save PIL image to buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return ImageArtifact(value=buffer.getvalue(), format="png", height=self.height, width=self.width)

    def try_image_outpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: Optional[list[str]] = None,
    ) -> ImageArtifact:
        raise NotImplementedError(f"Image outpainting is not supported by {self.__class__.__name__}")

    def try_image_inpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: Optional[list[str]] = None,
    ) -> ImageArtifact:
        raise NotImplementedError(f"Image inpainting is not supported by {self.__class__.__name__}")

    def _make_additional_params(self, negative_prompts: list[str]) -> dict[str, any]:
        additional_params = {}
        if negative_prompts:
            additional_params["negative_prompt"] = ", ".join(negative_prompts)

        if self.seed is not None:
            additional_params["generator"] = [torch.Generator(device=self.device).manual_seed(self.seed)]

        if self.guidance_scale is not None:
            additional_params["guidance_scale"] = self.guidance_scale

        if self.steps is not None:
            additional_params["num_inference_steps"] = self.steps

        if self.strength is not None:
            additional_params["strength"] = self.strength

        return additional_params
