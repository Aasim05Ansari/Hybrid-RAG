from typing import Any

from app.indexing.bm25_store import BM25Store
from app.retrieval.models import RetrievalResult


class SparseRetriever:

    def __init__(
        self,
        bm25_store: BM25Store,
    ):
        self.bm25_store = bm25_store

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

        results = self.bm25_store.search(
            query=query,
            top_k=top_k,
        )

        retrieval_results = []

        for result in results:

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=result["id"],
                    content=result["document"],
                    metadata=result["metadata"],
                    score=result["score"],
                )
            )

        return retrieval_results
