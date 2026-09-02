import re
from dataclasses import dataclass


@dataclass
class AnswerCompleteness:
    score: float
    covered_terms: list[str]
    missing_terms: list[str]


class AnswerCompletenessEvaluator:

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    SEMANTIC_GROUPS = {
        "allowance": {
            "allowance",
            "allowances",
            "days",
            "day",
            "entitlement",
            "entitled",
        },
        "cost": {
            "cost",
            "price",
            "fee",
            "amount",
        },
        "duration": {
            "duration",
            "days",
            "day",
            "weeks",
            "week",
            "months",
            "month",
            "time",
        },
        "location": {
            "location",
            "place",
            "address",
        },
        "number": {
            "number",
            "count",
            "amount",
            "total",
        },
    }

    def evaluate(
        self,
        question: str,
        answer: str,
    ) -> AnswerCompleteness:

        if not question.strip():
            raise ValueError("question cannot be empty")

        question_terms = self._terms(question)

        if not answer.strip():
            return AnswerCompleteness(
                score=0.0,
                covered_terms=[],
                missing_terms=question_terms,
            )

        answer_terms = set(self._terms(answer))

        if not question_terms:
            return AnswerCompleteness(
                score=1.0,
                covered_terms=[],
                missing_terms=[],
            )

        covered = []
        missing = []

        for term in question_terms:
            if self._term_is_covered(term, answer_terms):
                covered.append(term)
            else:
                missing.append(term)

        score = len(covered) / len(question_terms)

        return AnswerCompleteness(
            score=score,
            covered_terms=covered,
            missing_terms=missing,
        )

    @classmethod
    def _term_is_covered(
        cls,
        question_term: str,
        answer_terms: set[str],
    ) -> bool:

        if question_term in answer_terms:
            return True

        for group in cls.SEMANTIC_GROUPS.values():
            if question_term in group:
                if group.intersection(answer_terms):
                    return True

        return False

    @classmethod
    def _terms(cls, text: str) -> list[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())

        return [
            token
            for token in tokens
            if token not in cls.STOP_WORDS
        ]
