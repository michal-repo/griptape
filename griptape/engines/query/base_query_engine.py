from abc import ABC, abstractmethod
from typing import Optional
from attr import define
from griptape.artifacts import BaseArtifact, ListArtifact
from griptape.rules import Ruleset


@define
class BaseQueryEngine(ABC):
    @abstractmethod
    def query(
        self, query: str, namespace: Optional[str] = None, rulesets: Optional[list[Ruleset]] = None, **kwargs
    ) -> BaseArtifact:
        ...

    @abstractmethod
    def load_artifacts(self, namespace: str) -> ListArtifact:
        ...

    @abstractmethod
    def upsert_text_artifact(self, artifact: BaseArtifact, namespace: Optional[str] = None) -> str:
        ...

    @abstractmethod
    def upsert_text_artifacts(self, artifacts: list[BaseArtifact], namespace: str) -> None:
        ...
