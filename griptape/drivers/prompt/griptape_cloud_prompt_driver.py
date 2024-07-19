from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from attrs import Factory, define, field
from requests import post

from griptape.common import (
    DeltaMessage,
    Message,
    PromptStack,
)
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import SimpleTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griptape.tokenizers.base_tokenizer import BaseTokenizer


@define
class GriptapeCloudPromptDriver(BasePromptDriver):
    """Griptape Cloud Prompt Driver.

    Attributes:
        api_key: API Key for Griptape Cloud.
        base_url: Base URL for Griptape Cloud.
        headers: Headers for Griptape Cloud.
    """

    model: str = field(default="auto", kw_only=True)
    api_key: str = field(default=Factory(lambda: os.getenv("GT_CLOUD_API_KEY")), kw_only=True)
    base_url: str = field(default="https://cloud.griptape.ai", kw_only=True)
    headers: dict = field(
        default=Factory(lambda self: {"Authorization": f"Bearer {self.api_key}"}, takes_self=True),
        kw_only=True,
    )
    tokenizer: BaseTokenizer = field(
        default=Factory(
            lambda self: SimpleTokenizer(
                characters_per_token=4,
                max_input_tokens=2000,
                max_output_tokens=self.max_tokens,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )

    def try_run(self, prompt_stack: PromptStack) -> Message:
        url = urljoin(self.base_url.strip("/"), "/api/drivers/prompt")

        response = post(
            url,
            json=self._base_params(prompt_stack),
            headers=self.headers,
        )
        response.raise_for_status()
        response_json = response.json()

        return Message.from_dict(response_json)

    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        url = urljoin(self.base_url.strip("/"), "/api/drivers/prompt-stream")

        response = post(
            url,
            json=self._base_params(prompt_stack),
            stream=True,
            headers=self.headers,
        )
        response.raise_for_status()

        full_chunk = b""
        for chunk in response.iter_content():
            if chunk:
                full_chunk += chunk
                try:
                    decoded_chunk = full_chunk.decode("utf-8")
                    parsed_chunk = json.loads(decoded_chunk)
                    full_chunk = b""
                except json.JSONDecodeError:
                    continue
                yield DeltaMessage.from_dict(parsed_chunk)

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        messages = self._prompt_stack_to_messages(prompt_stack)

        return {
            "messages": messages,
            "params": {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

    def _prompt_stack_to_messages(self, prompt_stack: PromptStack) -> list[dict]:
        return [message.to_dict() for message in prompt_stack.messages]
