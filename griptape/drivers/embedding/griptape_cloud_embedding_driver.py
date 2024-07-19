from __future__ import annotations

import os
from urllib.parse import urljoin

from attrs import Factory, define, field
from requests import post

from griptape.drivers import BaseEmbeddingDriver


@define
class GriptapeCloudEmbeddingDriver(BaseEmbeddingDriver):
    """Griptape Cloud Embedding Driver.

    Attributes:
        api_key: API Key for Griptape Cloud.
        base_url: Base URL for Griptape Cloud.
        headers: Headers for Griptape Cloud.
    """

    model: str = field(kw_only=True, metadata={"serializable": True})
    base_url: str = field(default="https://cloud.griptape.ai", kw_only=True)
    api_key: str = field(default=Factory(lambda: os.getenv("GT_CLOUD_API_KEY")), kw_only=True)
    headers: dict = field(
        default=Factory(lambda self: {"Authorization": f"Bearer {self.api_key}"}, takes_self=True),
        kw_only=True,
    )

    def try_embed_chunk(self, chunk: str) -> list[float]:
        url = urljoin(self.base_url.strip("/"), "/api/drivers/embedding")

        response = post(
            url,
            json=self._params(chunk),
            headers=self.headers,
        )
        response.raise_for_status()
        response_json = response.json()

        return response_json

    def _params(self, chunk: str) -> dict:
        return {"input": chunk, "params": {"model": self.model}}
