from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NoReturn, Optional

import psycopg2
from attrs import Factory, define, field

from griptape.drivers import BaseEmbeddingDriver, BaseVectorStoreDriver, DummyEmbeddingDriver

if TYPE_CHECKING:
    from griptape.artifacts import ListArtifact, TextArtifact

logger = logging.getLogger(__name__)


@define
class AidbVectorStoreDriver(BaseVectorStoreDriver):
    """A vector store driver for AIDB.

    Attributes:
        dbname: DB Name
        user: DB User
        password: DB Password
        host: DB Hostname
        port: DB Port
    """

    dbname: str = field(kw_only=True, metadata={"serializable": True})
    user: str = field(kw_only=True, metadata={"serializable": True})
    password: str = field(kw_only=True, metadata={"serializable": True})
    host: str = field(kw_only=True, metadata={"serializable": True})
    port: str = field(kw_only=True, metadata={"serializable": True})
    embedding_driver: BaseEmbeddingDriver = field(
        default=Factory(lambda: DummyEmbeddingDriver()),
        metadata={"serializable": True},
        kw_only=True,
        init=False,
    )

    engine = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )

    def upsert_vector(
        self,
        vector: list[float],
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector upsert.")

    def upsert_text_artifact(
        self,
        artifact: TextArtifact,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        vector_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support text artifact upsert.")

    def upsert_text(
        self,
        string: str,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support text upsert.")

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> BaseVectorStoreDriver.Entry:
        raise NotImplementedError(f"{self.__class__.__name__} does not support entry loading.")

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support entry loading.")

    def load_artifacts(self, *, namespace: Optional[str] = None) -> ListArtifact:
        raise NotImplementedError(f"{self.__class__.__name__} does not support Artifact loading.")

    def query(
        self,
        query: str,
        *,
        count: Optional[int] = 2,
        namespace: Optional[str] = None,
        include_vectors: Optional[bool] = None,
        distance_metric: Optional[str] = None,
        # AidbVectorStoreDriver-specific params:
        retriever_name: Optional[str] = "img_embeddings",  # noqa: A002
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a query on AIDB."""
        cur = self.engine.cursor()
        try:
            cur.execute(f"""SELECT data from aidb.retrieve('{query}', {count}, {retriever_name});""")
            results = cur.fetchall()
            query_results = [result[0] for result in results]

        except Exception as e:
            logger.error("An error occurred: " + str(e))
        finally:
            cur.close()

        entries = []
        if query_results:
            for result in query_results:
                img_id = eval(result)["img_id"]
                logger.info(f"img_id: {img_id}")
                entries.append({"id": img_id})
        entry_list = [BaseVectorStoreDriver.Entry.from_dict(entry) for entry in entries]
        return entry_list

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")
