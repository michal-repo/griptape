import boto3
import pytest
from griptape.drivers.prompt_model.sagemaker_jumpstart_llama3_instruct_prompt_model_driver import (
    SageMakerJumpStartLlama3InstructPromptModelDriver,
)
from griptape.utils import PromptStack
from griptape.drivers import AmazonSageMakerPromptDriver


class TestSageMakerJumpStartLlama3InstructPromptModelDriver:
    @pytest.fixture
    def driver(self):
        return AmazonSageMakerPromptDriver(
            endpoint="endpoint-name",
            model="inference-component-name",
            session=boto3.Session(region_name="us-east-1"),
            prompt_model_driver=SageMakerJumpStartLlama3InstructPromptModelDriver(),
            temperature=0.12345,
        ).prompt_model_driver

    @pytest.fixture
    def stack(self):
        stack = PromptStack()

        stack.add_system_input("foo")
        stack.add_user_input("bar")

        return stack

    def test_init(self, driver):
        assert driver.prompt_driver is not None

    def test_prompt_stack_to_model_input(self, driver, stack):
        model_input = driver.prompt_stack_to_model_input(stack)

        assert isinstance(model_input, str)
        assert model_input == (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\nfoo<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nbar<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def test_prompt_stack_to_model_params(self, driver, stack):
        assert driver.prompt_stack_to_model_params(stack)["max_new_tokens"] == 7991
        assert driver.prompt_stack_to_model_params(stack)["temperature"] == 0.12345

    def test_process_output(self, driver, stack):
        assert driver.process_output({"generated_text": "foobar"}).value == "foobar"

    def test_tokenizer_max_model_length(self, driver):
        assert driver.tokenizer.tokenizer.model_max_length == 8000
