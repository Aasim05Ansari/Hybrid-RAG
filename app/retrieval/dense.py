
from app.embeddings.embedder import OpenAIEmbeddingProvider
from app.indexing.vector_store import VectorStore
from app.retrieval.models import RetrievalResult




class DenseRetriever:

    def __init__(
        self,
        embedding_provider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_embedding = (
            self.embedding_provider.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        retrieval_results = []

        for result in results:

            distance = result["distance"]

            # Chroma is configured for cosine distance.
            # Convert distance into similarity.
            score = 1.0 - distance

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=result["id"],
                    content=result["document"],
                    metadata=result["metadata"],
                    score=score,
                )
            )

        return retrieval_results
