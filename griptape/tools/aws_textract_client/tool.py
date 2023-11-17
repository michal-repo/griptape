from __future__ import annotations
from typing import Any
from schema import Schema
import json
from attr import define, field, Factory
from griptape.artifacts import TextArtifact, ErrorArtifact, InfoArtifact
from griptape.utils.decorators import activity
from griptape.tools import BaseAwsClient


@define
class AwsTextractClient(BaseAwsClient):
    textract_client: Any = field(
        default=Factory(lambda self: self.session.client("textract"), takes_self=True), kw_only=True
    )

    @activity(
        config={
            "description": "Can be use to analyze an expense",
            "schema": Schema({"memory_name": str, "artifact_namespace": str}),
        }
    )
    def analyze_expense(self, params: dict) -> TextArtifact | ErrorArtifact | InfoArtifact:
        memory = self.find_input_memory(params["values"]["memory_name"])
        artifact_namespace = params["values"]["artifact_namespace"]

        if memory:
            artifacts = memory.load_artifacts(artifact_namespace)
            if len(artifacts) == 0:
                return ErrorArtifact("no artifacts found")
            elif len(artifacts) == 1:
                try:
                    result = self._analyze_expense(artifacts.value[0].value)

                    return TextArtifact(result)
                except Exception as e:
                    return ErrorArtifact(f"error uploading objects to the bucket: {e}")
        else:
            return ErrorArtifact("memory not found")

    def _analyze_expense(self, document_bytes: bytes) -> dict:
        result = self.textract_client.analyze_expense(Document={"Bytes": document_bytes})
        document = result["ExpenseDocuments"][0]
        summary_fields = self._get_summary_fields(document)
        line_items = self._get_line_items(document)

        return {"summary_fields": summary_fields, "line_items": line_items}

    def _get_summary_fields(self, document: dict) -> list[dict]:
        summary_fields = document["SummaryFields"]

        acc = []
        for summary_field in summary_fields:
            field_type = summary_field["Type"]["Text"]
            value = summary_field["ValueDetection"]["Text"]
            label = summary_field["LabelDetection"]["Text"] if "LabelDetection" in summary_field else ""
            acc.append({"text": field_type, "value": value, "label": label})

        return acc

    def _get_line_items(self, document: dict) -> list[dict]:
        line_item_groups = document["LineItemGroups"]

        items = []

        for line_item_group in line_item_groups:
            result = []
            for line_item in line_item_group["LineItems"]:
                expense_fields = line_item["LineItemExpenseFields"]
                line_item = {}
                for expense_field in expense_fields:
                    field_type = expense_field["Type"]["Text"]
                    value = expense_field["ValueDetection"]["Text"]
                    line_item[field_type] = value
                result.append(line_item)
            items.append(result)
        return items
