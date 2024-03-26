from __future__ import annotations
from abc import ABC
from attr import define, field, Factory
from griptape.artifacts import TextArtifact
from griptape.tokenizers import BaseTokenizer, OpenAiTokenizer


@define
class FixedTokenChunker(ABC):
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda: OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL)), kw_only=True
    )
    max_tokens: int = field(
        default=Factory(lambda self: self.tokenizer.max_input_tokens, takes_self=True), kw_only=True
    )
    overlap: int = field(default=0, kw_only=True)

    def chunk(self, text: TextArtifact | str) -> list[TextArtifact]:
        text = text.value if isinstance(text, TextArtifact) else text

        return [TextArtifact(c) for c in self._brute_force(text)]

    def _brute_force(self, text: str) -> list[str]:
        chunks = []
        start = 0
        next_start = 0
        for end in range(1, len(text) + 1):
            chunk = text[start:end]
            token_count = self.tokenizer.count_tokens(chunk)

            if end == len(text):
                chunks.append(chunk)
            elif token_count == self.max_tokens:
                chunks.append(chunk)
                next_start = end
                overlap = 0
                while overlap < self.overlap:
                    next_start -= 1
                    overlap = self.tokenizer.count_tokens(text[next_start:end])
                start = next_start

        return chunks
