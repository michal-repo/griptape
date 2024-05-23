from __future__ import annotations
from attr import define, field
from griptape.artifacts import TextArtifact
from griptape.utils import PromptStack, import_optional_dependency
from griptape.drivers import BasePromptModelDriver
from griptape.tokenizers import HuggingFaceTokenizer


@define
class SageMakerJumpStartLlama3InstructPromptModelDriver(BasePromptModelDriver):
    # Default context length for all Llama 3 models is 8K as per https://huggingface.co/blog/llama3
    DEFAULT_MAX_TOKENS = 8000

    _tokenizer: HuggingFaceTokenizer = field(default=None, kw_only=True)

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        if self._tokenizer is None:
            self._tokenizer = HuggingFaceTokenizer(
                tokenizer=import_optional_dependency("transformers").PreTrainedTokenizerFast.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct", model_max_length=self.DEFAULT_MAX_TOKENS
                ),
                max_output_tokens=self.max_tokens or self.DEFAULT_MAX_TOKENS,
            )
        return self._tokenizer

    def prompt_stack_to_model_input(self, prompt_stack: PromptStack) -> str:
        # This input format is specific to the Llama 3 Instruct model prompt format.
        # For more details see: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3#meta-llama-3-instruct
        return "".join(
            [
                "<|begin_of_text|>",
                *[
                    f"<|start_header_id|>{i.role}<|end_header_id|>\n\n{i.content}<|eot_id|>"
                    for i in prompt_stack.inputs
                ],
                f"<|start_header_id|>{PromptStack.ASSISTANT_ROLE}<|end_header_id|>\n\n",
            ]
        )

    def prompt_stack_to_model_params(self, prompt_stack: PromptStack) -> dict:
        prompt = self.prompt_driver.prompt_stack_to_string(prompt_stack)

        return {
            "max_new_tokens": self.prompt_driver.max_output_tokens(prompt),
            "temperature": self.prompt_driver.temperature,
            # This stop parameter is specific to the Llama 3 Instruct model prompt format.
            # docs: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3#meta-llama-3-instruct
            "stop": "<|eot_id|>",
        }

    def process_output(self, output: dict | list[dict] | str | bytes) -> TextArtifact:
        # This output format is specific to the Llama 3 Instruct models when deployed via SageMaker JumpStart.
        if isinstance(output, dict):
            return TextArtifact(output["generated_text"])
        else:
            raise ValueError("output must be an instance of 'dict'")
