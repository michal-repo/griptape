from typing import Optional

from attr import field, define

from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseImageGenerationDriver, BaseImageModificationDriver
from griptape.rules import Rule, Ruleset


@define
class ImageModificationEngine:
    image_modification_driver: BaseImageModificationDriver = field(kw_only=True)

    def modify_image(
        self,
        input_image: ImageArtifact,
        prompts: list[str],
        mask_image: Optional[ImageArtifact] = None,
        negative_prompts: Optional[list[str]] = None,
        rulesets: Optional[list[Ruleset]] = None,
        negative_rulesets: Optional[list[Ruleset]] = None,
    ):
        if not negative_prompts:
            negative_prompts = []

        if rulesets is not None:
            for ruleset in rulesets:
                prompts += [rule.value for rule in ruleset.rules]

        if negative_rulesets is not None:
            for negative_ruleset in negative_rulesets:
                negative_prompts += [rule.value for rule in negative_ruleset.rules]

        return self.image_modification_driver.modify_image(
            input_image, prompts, mask_image=mask_image, negative_prompts=negative_prompts
        )
