from __future__ import annotations

from attrs import define, field

from griptape.drivers import BaseEmbeddingDriver
from tests.mocks.mock_tokenizer import MockTokenizer


@define
class MockEmbeddingDriver(BaseEmbeddingDriver):
    model: str = field(default="foo", kw_only=True)
    dimensions: int = field(default=42, kw_only=True)
    max_attempts: int = field(default=1, kw_only=True)
    tokenizer: MockTokenizer = field(factory=lambda: MockTokenizer(model="foo bar"), kw_only=True)

    def try_embed_chunk(self, chunk: str) -> list[float]:
        return [0, 1]
