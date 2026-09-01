from math import sqrt

from app.embeddings.embedder import EmbeddingProvider
from app.ingestion.metadata import Chunk
from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import VectorStore


class Indexer:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        duplicate_similarity_threshold: float = 0.95,
    ):
        if not 0 <= duplicate_similarity_threshold <= 1:
            raise ValueError(
                "duplicate_similarity_threshold "
                "must be between 0 and 1"
            )

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.duplicate_similarity_threshold = (
            duplicate_similarity_threshold
        )

    def index(
        self,
        chunks: list[Chunk],
    ) -> dict[str, list[str]]:

        if not chunks:
            return {
                "indexed": [],
                "duplicates": [],
            }

        documents = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_provider.embed_documents(
                documents
            )
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match "
                "number of chunks"
            )

        indexed_chunks = []
        indexed_embeddings = []
        duplicate_ids = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            if self._is_duplicate(
                embedding,
                indexed_embeddings,
            ):
                duplicate_ids.append(
                    chunk.chunk_id
                )
                continue

            indexed_chunks.append(chunk)
            indexed_embeddings.append(embedding)

        if not indexed_chunks:
            return {
                "indexed": [],
                "duplicates": duplicate_ids,
            }

        ids = [
            chunk.chunk_id
            for chunk in indexed_chunks
        ]

        indexed_documents = [
            chunk.content
            for chunk in indexed_chunks
        ]

        metadatas = [
            chunk.metadata
            for chunk in indexed_chunks
        ]

        self.vector_store.add(
            ids=ids,
            embeddings=indexed_embeddings,
            documents=indexed_documents,
            metadatas=metadatas,
        )

        self.bm25_store.add(
            ids=ids,
            documents=indexed_documents,
            metadatas=metadatas,
        )

        return {
            "indexed": ids,
            "duplicates": duplicate_ids,
        }

    def _is_duplicate(
        self,
        embedding: list[float],
        batch_embeddings: list[list[float]],
    ) -> bool:

        for existing_embedding in batch_embeddings:

            similarity = self.cosine_similarity(
                embedding,
                existing_embedding,
            )

            if (
                similarity
                > self.duplicate_similarity_threshold
            ):
                return True

        existing_results = (
            self.vector_store.search(
                query_embedding=embedding,
                top_k=1,
            )
        )

        if not existing_results:
            return False

        nearest = existing_results[0]

        distance = nearest["distance"]

        similarity = 1.0 - distance

        return (
            similarity
            > self.duplicate_similarity_threshold
        )

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding vectors must have "
                "the same dimension"
            )

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )
