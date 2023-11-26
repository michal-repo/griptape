from abc import abstractmethod
from typing import Optional
from attr import define
from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseImageDriver


@define
class BaseImageModificationDriver(BaseImageDriver):
    def modify_image(
        self,
        input_image: ImageArtifact,
        prompts: list[str],
        mask_image: Optional[ImageArtifact] = None,
        negative_prompts: Optional[list[str]] = None,
    ) -> ImageArtifact:
        for attempt in self.retrying():
            with attempt:
                return self.try_modify_image(
                    input_image, prompts=prompts, mask_image=mask_image, negative_prompts=negative_prompts
                )

    @abstractmethod
    def try_modify_image(
        self,
        image: ImageArtifact,
        prompts: list[str],
        mask_image: Optional[ImageArtifact] = None,
        negative_prompts: Optional[list[str]] = None,
    ) -> ImageArtifact:
        ...
