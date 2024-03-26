import pytest
from griptape.chunkers.fixed_token_chunker import FixedTokenChunker
from griptape.tokenizers.simple_tokenizer import SimpleTokenizer


class TestFixedTokenChunker:
    @pytest.fixture
    def tokenizer(self):
        return SimpleTokenizer(max_input_tokens=1, max_output_tokens=1, characters_per_token=1)

    @pytest.mark.parametrize(
        "max_tokens, overlap, expected_chunks",
        [
            (1, 0, ["a", "b", "c", "d", "e"]),
            (2, 0, ["ab", "cd", "e"]),
            (2, 1, ["ab", "bc", "cd", "de"]),
            (3, 0, ["abc", "de"]),
            (3, 1, ["abc", "cde"]),
            (3, 2, ["abc", "bcd", "cde"]),
            (4, 0, ["abcd", "e"]),
            (4, 1, ["abcd", "de"]),
            (4, 2, ["abcd", "cde"]),
            (5, 0, ["abcde"]),
            (5, 1, ["abcde"]),
            (5, 2, ["abcde"]),
            # Raises -> (1, 1, ["a", "b", "c", "d", "e"]),
        ],
    )
    def test_chunk(self, tokenizer, max_tokens, overlap, expected_chunks):
        chunker = FixedTokenChunker(tokenizer=tokenizer, max_tokens=max_tokens, overlap=overlap)
        text = "abcde"
        text_artifacts = chunker.chunk(text)

        chunks = [text_artifact.value for text_artifact in text_artifacts]

        assert chunks == expected_chunks
