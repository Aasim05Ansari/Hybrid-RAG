from dataclasses import dataclass


@dataclass
class CompositeConfidence:
    score: float
    retrieval_confidence: float
    citation_accuracy: float
    citation_coverage: float
    answer_completeness: float


class CompositeConfidenceCalculator:

    RETRIEVAL_WEIGHT = 0.30
    CITATION_ACCURACY_WEIGHT = 0.25
    CITATION_COVERAGE_WEIGHT = 0.25
    COMPLETENESS_WEIGHT = 0.20

    CITATION_ACCURACY_FLOOR = 0.50
    CITATION_ACCURACY_CAP = 0.49

    def calculate(
        self,
        retrieval_confidence: float,
        citation_accuracy: float,
        citation_coverage: float,
        answer_completeness: float,
    ) -> CompositeConfidence:

        values = {
            "retrieval_confidence": retrieval_confidence,
            "citation_accuracy": citation_accuracy,
            "citation_coverage": citation_coverage,
            "answer_completeness": answer_completeness,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        score = (
            self.RETRIEVAL_WEIGHT * retrieval_confidence
            + self.CITATION_ACCURACY_WEIGHT * citation_accuracy
            + self.CITATION_COVERAGE_WEIGHT * citation_coverage
            + self.COMPLETENESS_WEIGHT * answer_completeness
        )

        # Prevent strong retrieval or completeness scores from
        # masking seriously unreliable citations.
        if citation_accuracy < self.CITATION_ACCURACY_FLOOR:
            score = min(score, self.CITATION_ACCURACY_CAP)

        score = max(0.0, min(1.0, score))

        return CompositeConfidence(
            score=score,
            retrieval_confidence=retrieval_confidence,
            citation_accuracy=citation_accuracy,
            citation_coverage=citation_coverage,
            answer_completeness=answer_completeness,
        )
