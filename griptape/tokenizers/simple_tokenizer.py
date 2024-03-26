from __future__ import annotations
from attr import define, field
from griptape.tokenizers import BaseTokenizer


@define()
class SimpleTokenizer(BaseTokenizer):
    model: str = field(kw_only=True, init=False)
    characters_per_token: int = field(kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.max_input_tokens is None:
            raise ValueError(
                "SimpleTokenizer requires max_input_tokens to be set explicitly as has no model to derive it from."
            )

        if self.max_output_tokens is None:
            raise ValueError(
                "SimpleTokenizer requires max_output_tokens to be set explicitly as has no model to derive it from."
            )

    def count_tokens(self, text: str | list) -> int:
        if isinstance(text, str):
            num_tokens = (len(text) + self.characters_per_token - 1) // self.characters_per_token

            return num_tokens
        else:
            raise ValueError("Text must be a string.")
