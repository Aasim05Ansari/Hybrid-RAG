from dataclasses import dataclass

from app.retrieval.models import RetrievalResult


@dataclass
class RetrievalConfidence:
    score: float
    result_count: int
    top_score: float
    score_spread: float


class RetrievalConfidenceCalculator:

    def calculate(
        self,
        results: list[RetrievalResult],
    ) -> RetrievalConfidence:

        if not results:
            return RetrievalConfidence(
                score=0.0,
                result_count=0,
                top_score=0.0,
                score_spread=0.0,
            )

        scores = [result.score for result in results]

        top_score = max(scores)
        lowest_score = min(scores)

        score_spread = top_score - lowest_score

        # Scores from the current reranker are normalized between
        # 0 and 1, so clamp defensively.
        normalized_top_score = max(
            0.0,
            min(1.0, top_score),
        )

        # A larger score spread means the top result stands out
        # more clearly from weaker results.
        normalized_spread = max(
            0.0,
            min(1.0, score_spread),
        )

        # Reward having multiple retrieved results, but cap the
        # contribution at 5 results.
        result_factor = min(len(results), 5) / 5

        confidence = (
            0.6 * normalized_top_score
            + 0.3 * normalized_spread
            + 0.1 * result_factor
        )

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return RetrievalConfidence(
            score=confidence,
            result_count=len(results),
            top_score=top_score,
            score_spread=score_spread,
        )
