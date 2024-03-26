from __future__ import annotations
from abc import ABC
from typing import Optional
from attr import define, field, Factory
from griptape.artifacts import TextArtifact
from griptape.chunkers import ChunkSeparator
from griptape.tokenizers import BaseTokenizer, OpenAiTokenizer


@define
class FixedCharacterChunker(ABC):
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda: OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL)), kw_only=True
    )
    # Max characters should serve as an upperbound for the number of tokens, since
    # it is reasobale to assume that the number of tokens will be less than the number
    # of characters for a given text.
    max_characters: int = field(
        default=Factory(lambda self: self.tokenizer.max_input_tokens, takes_self=True), kw_only=True
    )
    overlap: int = field(default=0, kw_only=True)
    count: int = field(default=0, kw_only=True)

    def __attrs_post_init__(self):
        if self.max_characters <= 0:
            raise ValueError("max_characters must be greater than 0")
        if self.overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        if self.overlap >= self.max_characters:
            raise ValueError("overlap must be less than max_characters")

    def chunk(self, text: TextArtifact | str) -> list[TextArtifact]:
        text = text.value if isinstance(text, TextArtifact) else text
        stride = self.max_characters - self.overlap
        return [
            TextArtifact(text[start:end])
            for start in range(0, len(text) - self.overlap, stride)
            for end in [start + self.max_characters]
        ]
