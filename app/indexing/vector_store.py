import chromadb
from app.config.settings import settings
from abc import ABC, abstractmethod
from typing import Any



class VectorStore(ABC):

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents and their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        raise NotImplementedError
    
class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_directory = (
            persist_directory
            or settings.chroma_persist_dir
        )

        self.collection_name = (
            collection_name
            or settings.collection_name
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:

        if not ids:
            return

        if not (
            len(ids)
            == len(embeddings)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, embeddings, documents, and "
                "metadatas must have the same length"
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        results = []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for index, chunk_id in enumerate(ids):

            results.append(
                {
                    "id": chunk_id,
                    "document": documents[index],
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )

        return results