from __future__ import annotations
import json
from typing import Optional
from attr import define, field
from griptape import utils
from griptape.artifacts import JsonArtifact
from griptape.drivers import BaseEmbeddingDriver
from griptape.loaders import BaseLoader


@define
class JsonLoader(BaseLoader):
    embedding_driver: Optional[BaseEmbeddingDriver] = field(default=None, kw_only=True)
    delimiter: str = field(default=",", kw_only=True)

    def load(self, filename: str) -> list[JsonArtifact]:
        return self._load_file(filename)

    def load_collection(self, filenames: list[str]) -> dict[str, list[JsonArtifact]]:
        return utils.execute_futures_dict(
            {
                utils.str_to_hash(filename): self.futures_executor.submit(self._load_file, filename)
                for filename in filenames
            }
        )

    def _load_file(self, filename: str) -> list[JsonArtifact]:
        with open(filename, "r", encoding="utf-8") as json_file:
            try:
                reader = json.load(json_file)
                if isinstance(reader, dict):
                    chunks = [JsonArtifact(reader)]
                elif isinstance(reader, list):
                    chunks = [JsonArtifact(row) for row in reader]
                else:
                    raise ValueError(f"Invalid JSON file: {filename}")

                if self.embedding_driver:
                    for chunk in chunks:
                        chunk.generate_embedding(self.embedding_driver)

                artifacts = []
                for chunk in chunks:
                    artifacts.append(chunk)

                return artifacts
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON file: {filename}")
