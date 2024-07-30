from unittest.mock import Mock

import pytest

from griptape.artifacts import ActionArtifact, ImageArtifact, ListArtifact, TextArtifact
from griptape.common import ActionCallDeltaMessageContent, PromptStack, TextDeltaMessageContent, ToolAction
from griptape.drivers import OpenAiChatPromptDriver
from griptape.tokenizers import OpenAiTokenizer
from tests.mocks.mock_tokenizer import MockTokenizer
from tests.mocks.mock_tool.tool import MockTool


class TestOpenAiChatPromptDriverFixtureMixin:
    OPENAI_TOOLS = [
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_error",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_exception",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_list_output",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_no_schema",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_str_output",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_without_default_memory",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
    ]

    @pytest.fixture()
    def mock_chat_completion_create(self, mocker):
        mock_chat_create = mocker.patch("openai.OpenAI").return_value.chat.completions.create
        mock_function = Mock(arguments='{"foo": "bar"}', id="mock-id")
        mock_function.name = "MockTool_test"
        mock_chat_create.return_value = Mock(
            headers={},
            choices=[
                Mock(message=Mock(content="model-output", tool_calls=[Mock(id="mock-id", function=mock_function)]))
            ],
            usage=Mock(prompt_tokens=5, completion_tokens=10),
        )

        return mock_chat_create

    @pytest.fixture()
    def mock_chat_completion_stream_create(self, mocker):
        mock_chat_create = mocker.patch("openai.OpenAI").return_value.chat.completions.create
        mock_tool_call_delta_header = Mock()
        mock_tool_call_delta_header.name = "MockTool_test"
        mock_tool_call_delta_body = Mock(arguments='{"foo": "bar"}')
        mock_tool_call_delta_body.name = None

        mock_chat_create.return_value = iter(
            [
                Mock(choices=[Mock(delta=Mock(content="model-output", tool_calls=None))], usage=None),
                Mock(
                    choices=[
                        Mock(
                            delta=Mock(
                                content=None,
                                tool_calls=[Mock(index=0, id="mock-id", function=mock_tool_call_delta_header)],
                            )
                        )
                    ],
                    usage=None,
                ),
                Mock(
                    choices=[
                        Mock(
                            delta=Mock(
                                content=None, tool_calls=[Mock(index=0, id=None, function=mock_tool_call_delta_body)]
                            )
                        )
                    ],
                    usage=None,
                ),
                Mock(choices=None, usage=Mock(prompt_tokens=5, completion_tokens=10)),
                Mock(choices=[Mock(delta=Mock(content=None, tool_calls=None))], usage=None),
            ]
        )
        return mock_chat_create

    @pytest.fixture()
    def prompt_stack(self):
        prompt_stack = PromptStack()
        prompt_stack.tools = [MockTool()]
        prompt_stack.add_system_message("system-input")
        prompt_stack.add_user_message("user-input")
        prompt_stack.add_user_message(
            ListArtifact(
                [TextArtifact("user-input"), ImageArtifact(value=b"image-data", format="png", width=100, height=100)]
            )
        )
        prompt_stack.add_assistant_message("assistant-input")
        prompt_stack.add_assistant_message(
            ListArtifact(
                [
                    TextArtifact("thought"),
                    ActionArtifact(ToolAction(tag="MockTool_test", name="MockTool", path="test", input={"foo": "bar"})),
                ]
            )
        )
        prompt_stack.add_user_message(
            ListArtifact(
                [
                    TextArtifact("keep-going"),
                    ActionArtifact(
                        ToolAction(
                            tag="MockTool_test",
                            name="MockTool",
                            path="test",
                            input={"foo": "bar"},
                            output=TextArtifact("tool-output"),
                        )
                    ),
                ]
            )
        )
        return prompt_stack

    @pytest.fixture()
    def messages(self):
        return [
            {"role": "system", "content": "system-input"},
            {"role": "user", "content": "user-input"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "user-input"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2UtZGF0YQ=="}},
                ],
            },
            {"role": "assistant", "content": "assistant-input"},
            {
                "content": [{"text": "thought", "type": "text"}],
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"arguments": '{"foo": "bar"}', "name": "MockTool_test"},
                        "id": "MockTool_test",
                        "type": "function",
                    }
                ],
            },
            {"content": "tool-output", "role": "tool", "tool_call_id": "MockTool_test"},
            {"content": "keep-going", "role": "user"},
        ]


class OpenAiApiResponseWithHeaders:
    def __init__(
        self,
        reset_requests_in=5,
        reset_requests_in_unit="s",
        reset_tokens_in=10,
        reset_tokens_in_unit="s",
        remaining_requests=123,
        remaining_tokens=234,
        limit_requests=345,
        limit_tokens=456,
    ) -> None:
        self.reset_requests_in = reset_requests_in
        self.reset_requests_in_unit = reset_requests_in_unit
        self.reset_tokens_in = reset_tokens_in
        self.reset_tokens_in_unit = reset_tokens_in_unit
        self.remaining_requests = remaining_requests
        self.remaining_tokens = remaining_tokens
        self.limit_requests = limit_requests
        self.limit_tokens = limit_tokens

    @property
    def headers(self):
        return {
            "x-ratelimit-reset-requests": f"{self.reset_requests_in}{self.reset_requests_in_unit}",
            "x-ratelimit-reset-tokens": f"{self.reset_tokens_in}{self.reset_tokens_in_unit}",
            "x-ratelimit-limit-requests": self.limit_requests,
            "x-ratelimit-remaining-requests": self.remaining_requests,
            "x-ratelimit-limit-tokens": self.limit_tokens,
            "x-ratelimit-remaining-tokens": self.remaining_tokens,
        }


class TestOpenAiChatPromptDriver(TestOpenAiChatPromptDriverFixtureMixin):
    def test_init(self):
        assert OpenAiChatPromptDriver(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_4_MODEL)

    @pytest.mark.parametrize("use_native_tools", [True, False])
    def test_try_run(self, mock_chat_completion_create, prompt_stack, messages, use_native_tools):
        # Given
        driver = OpenAiChatPromptDriver(
            model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL, use_native_tools=use_native_tools
        )

        # When
        message = driver.try_run(prompt_stack)

        # Then
        mock_chat_completion_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            user=driver.user,
            messages=messages,
            seed=driver.seed,
            **{"tools": self.OPENAI_TOOLS, "tool_choice": driver.tool_choice} if use_native_tools else {},
        )
        assert isinstance(message.value[0], TextArtifact)
        assert message.value[0].value == "model-output"
        assert isinstance(message.value[1], ActionArtifact)
        assert message.value[1].value.tag == "mock-id"
        assert message.value[1].value.name == "MockTool"
        assert message.value[1].value.path == "test"
        assert message.value[1].value.input == {"foo": "bar"}

    def test_try_run_response_format(self, mock_chat_completion_create, prompt_stack, messages):
        # Given
        driver = OpenAiChatPromptDriver(
            model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL, response_format="json_object", use_native_tools=False
        )

        # When
        message = driver.try_run(prompt_stack)

        # Then
        mock_chat_completion_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            user=driver.user,
            messages=[*messages, {"role": "system", "content": "Provide your response as a valid JSON object."}],
            seed=driver.seed,
            response_format={"type": "json_object"},
        )
        assert message.value[0].value == "model-output"
        assert message.usage.input_tokens == 5
        assert message.usage.output_tokens == 10

    @pytest.mark.parametrize("use_native_tools", [True, False])
    def test_try_stream_run(self, mock_chat_completion_stream_create, prompt_stack, messages, use_native_tools):
        # Given
        driver = OpenAiChatPromptDriver(
            model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL, stream=True, use_native_tools=use_native_tools
        )

        # When
        stream = driver.try_stream(prompt_stack)
        event = next(stream)

        # Then
        mock_chat_completion_stream_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            user=driver.user,
            stream=True,
            messages=messages,
            seed=driver.seed,
            stream_options={"include_usage": True},
            **{"tools": self.OPENAI_TOOLS, "tool_choice": driver.tool_choice} if use_native_tools else {},
        )

        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == "model-output"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.tag == "mock-id"
        assert event.content.name == "MockTool"
        assert event.content.path == "test"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.partial_input == '{"foo": "bar"}'

        event = next(stream)
        assert event.usage.input_tokens == 5
        assert event.usage.output_tokens == 10
        event = next(stream)
        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == ""

    def test_try_run_with_max_tokens(self, mock_chat_completion_create, prompt_stack, messages):
        # Given
        driver = OpenAiChatPromptDriver(
            model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL, max_tokens=1, use_native_tools=False
        )

        # When
        event = driver.try_run(prompt_stack)

        # Then
        mock_chat_completion_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            user=driver.user,
            messages=messages,
            max_tokens=1,
            seed=driver.seed,
        )
        assert event.value[0].value == "model-output"

    def test_try_run_throws_when_multiple_choices_returned(self, mock_chat_completion_create, prompt_stack):
        # Given
        driver = OpenAiChatPromptDriver(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL, api_key="api-key")
        mock_chat_completion_create.return_value.choices = [Mock(message=Mock(content="model-output"))] * 10

        # When
        with pytest.raises(Exception) as e:
            driver.try_run(prompt_stack)

        # Then
        assert e.value.args[0] == "Completion with more than one choice is not supported yet."

    def test_custom_tokenizer(self, mock_chat_completion_create, prompt_stack, messages):
        driver = OpenAiChatPromptDriver(
            model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL,
            tokenizer=MockTokenizer(model="mock-model", stop_sequences=["mock-stop"]),
            max_tokens=1,
            use_native_tools=False,
        )

        # When
        event = driver.try_run(prompt_stack)

        # Then
        mock_chat_completion_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            stop=driver.tokenizer.stop_sequences,
            user=driver.user,
            messages=messages,
            seed=driver.seed,
            max_tokens=1,
        )
        assert event.value[0].value == "model-output"
