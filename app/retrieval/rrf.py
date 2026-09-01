from app.retrieval.models import RetrievalResult


class RRFFusion:

    def __init__(
        self,
        k: int = 60,
    ):
        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        self.k = k

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        weights: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:

        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        if len(weights) != len(result_lists):
            raise ValueError(
                "Number of weights must match "
                "number of result lists"
            )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Weights cannot be negative"
            )

        scores: dict[str, float] = {}
        results_by_id: dict[str, RetrievalResult] = {}

        for results, weight in zip(
            result_lists,
            weights,
        ):

            for rank, result in enumerate(
                results,
                start=1,
            ):

                chunk_id = result.chunk_id

                rrf_score = (
                    weight
                    / (self.k + rank)
                )

                scores[chunk_id] = (
                    scores.get(
                        chunk_id,
                        0.0,
                    )
                    + rrf_score
                )

                results_by_id[chunk_id] = result

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        if top_k is not None:

            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0"
                )

            ranked_ids = ranked_ids[:top_k]

        fused_results = []

        for chunk_id in ranked_ids:

            original_result = results_by_id[
                chunk_id
            ]

            fused_results.append(
                RetrievalResult(
                    chunk_id=original_result.chunk_id,
                    content=original_result.content,
                    metadata=original_result.metadata,
                    score=scores[chunk_id],
                )
            )

        return fused_results
