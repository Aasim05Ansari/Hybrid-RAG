from dataclasses import dataclass


@dataclass
class AbstentionDecision:
    should_abstain: bool
    reason: str | None


class AbstentionPolicy:

    def __init__(
        self,
        min_composite_confidence: float = 0.50,
        min_retrieval_confidence: float = 0.25,
    ):
        if not 0.0 <= min_composite_confidence <= 1.0:
            raise ValueError(
                "min_composite_confidence must be between 0 and 1"
            )

        if not 0.0 <= min_retrieval_confidence <= 1.0:
            raise ValueError(
                "min_retrieval_confidence must be between 0 and 1"
            )

        self.min_composite_confidence = min_composite_confidence
        self.min_retrieval_confidence = min_retrieval_confidence

    def decide(
        self,
        composite_confidence: float,
        retrieval_confidence: float,
    ) -> AbstentionDecision:

        if not 0.0 <= composite_confidence <= 1.0:
            raise ValueError(
                "composite_confidence must be between 0 and 1"
            )

        if not 0.0 <= retrieval_confidence <= 1.0:
            raise ValueError(
                "retrieval_confidence must be between 0 and 1"
            )

        if retrieval_confidence < self.min_retrieval_confidence:
            return AbstentionDecision(
                should_abstain=True,
                reason="retrieval_confidence_too_low",
            )

        if composite_confidence < self.min_composite_confidence:
            return AbstentionDecision(
                should_abstain=True,
                reason="composite_confidence_too_low",
            )

        return AbstentionDecision(
            should_abstain=False,
            reason=None,
        )
