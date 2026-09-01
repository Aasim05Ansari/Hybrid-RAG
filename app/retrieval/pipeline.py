from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import Reranker


class RetrievalPipeline:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        final_top_k: int = 5,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.final_top_k = final_top_k

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than 0"
            )

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        # Stage 1:
        # Dense + Sparse + Weighted RRF
        # produces candidate documents.
        candidates = (
            self.hybrid_retriever.retrieve(
                query
            )
        )

        # Stage 2:
        # Rerank the candidates and keep
        # only the final top-K results.
        results = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=self.final_top_k,
        )

        return results
