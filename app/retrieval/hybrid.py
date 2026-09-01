from app.config.settings import settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.models import RetrievalResult
from app.retrieval.rrf import RRFFusion
from app.retrieval.sparse import SparseRetriever


class HybridRetriever:

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion: RRFFusion | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        fusion_top_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion or RRFFusion()

        self.dense_top_k = (
            dense_top_k
            if dense_top_k is not None
            else settings.dense_top_k
        )

        self.sparse_top_k = (
            sparse_top_k
            if sparse_top_k is not None
            else settings.sparse_top_k
        )

        self.fusion_top_k = (
            fusion_top_k
            if fusion_top_k is not None
            else settings.fusion_top_k
        )

        self.dense_weight = (
            dense_weight
            if dense_weight is not None
            else settings.dense_weight
        )

        self.sparse_weight = (
            sparse_weight
            if sparse_weight is not None
            else settings.sparse_weight
        )

        if self.dense_top_k <= 0:
            raise ValueError(
                "dense_top_k must be greater than 0"
            )

        if self.sparse_top_k <= 0:
            raise ValueError(
                "sparse_top_k must be greater than 0"
            )

        if self.fusion_top_k <= 0:
            raise ValueError(
                "fusion_top_k must be greater than 0"
            )

        if self.dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative"
            )

        if self.sparse_weight < 0:
            raise ValueError(
                "sparse_weight cannot be negative"
            )

        if (
            self.dense_weight
            + self.sparse_weight
            == 0
        ):
            raise ValueError(
                "At least one retrieval weight "
                "must be greater than 0"
            )

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=self.dense_top_k,
            )
        )

        sparse_results = (
            self.sparse_retriever.retrieve(
                query=query,
                top_k=self.sparse_top_k,
            )
        )

        return self.fusion.fuse(
            [
                dense_results,
                sparse_results,
            ],
            weights=[
                self.dense_weight,
                self.sparse_weight,
            ],
            top_k=self.fusion_top_k,
        )
