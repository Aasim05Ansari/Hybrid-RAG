from app.generation.abstention import AbstentionPolicy
from app.generation.citation_evaluator import CitationEvaluator
from app.generation.citation_verifier import CitationVerifier
from app.generation.citations import CitationExtractor
from app.generation.claim_mapping import ClaimCitationMapper
from app.generation.claim_verifier import (
    ClaimVerifier,
    ClaimVerification,
    VerificationStatus,
)
from app.generation.claims import ClaimExtractor
from app.generation.completeness import AnswerCompletenessEvaluator
from app.generation.composite_confidence import (
    CompositeConfidenceCalculator,
)
from app.generation.context import ContextBuilder
from app.generation.grounded_pipeline import GroundedGenerationPipeline
from app.generation.llm import LLMProvider
from app.generation.prompt import GroundedPromptBuilder
from app.generation.retrieval_confidence import (
    RetrievalConfidenceCalculator,
)
from app.retrieval.models import RetrievalResult


class FakeRetrievalPipeline:

    def __init__(self, results):
        self.results = results

    def retrieve(self, query):
        return self.results


class FakeLLMProvider(LLMProvider):

    def __init__(self, answer):
        self.answer = answer

    def generate(self, prompt):
        return self.answer


class FakeClaimVerifier(ClaimVerifier):

    def verify(self, claims):
        results = []

        for item in claims:
            results.append(
                ClaimVerification(
                    claim=item.claim,
                    citation=item.citation,
                    status=(
                        VerificationStatus.SUPPORTED
                        if item.evidence is not None
                        else VerificationStatus.INSUFFICIENT
                    ),
                    evidence=(
                        item.evidence.content
                        if item.evidence is not None
                        else None
                    ),
                )
            )

        return results


def make_result():
    return RetrievalResult(
        chunk_id="chunk-1",
        content=(
            "Employees are entitled to 24 days of annual leave "
            "per calendar year."
        ),
        metadata={
            "source": "employee_policy.pdf",
            "section": "Annual Leave",
        },
        score=0.9,
    )


def make_pipeline(answer):
    return GroundedGenerationPipeline(
        retrieval_pipeline=FakeRetrievalPipeline(
            [make_result()]
        ),
        context_builder=ContextBuilder(),
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=FakeLLMProvider(answer),
        citation_extractor=CitationExtractor(),
        citation_verifier=CitationVerifier(),
        claim_extractor=ClaimExtractor(),
        claim_mapper=ClaimCitationMapper(),
        claim_verifier=FakeClaimVerifier(),
        citation_evaluator=CitationEvaluator(),
        completeness_evaluator=AnswerCompletenessEvaluator(),
        retrieval_confidence_calculator=(
            RetrievalConfidenceCalculator()
        ),
        composite_confidence_calculator=(
            CompositeConfidenceCalculator()
        ),
        abstention_policy=AbstentionPolicy(),
    )


def test_end_to_end_grounded_answer():
    pipeline = make_pipeline(
        "Employees receive 24 days of annual leave [1]."
    )

    result = pipeline.run(
        "What is the annual leave allowance?"
    )

    assert result.answer == (
        "Employees receive 24 days of annual leave [1]."
    )

    assert result.abstention.should_abstain is False

    assert result.citation_verification.valid == [1]
    assert result.citation_verification.invalid == []

    assert len(result.claims) == 1
    assert result.claims[0].citation == 1

    assert result.claim_verifications[0].status == (
        VerificationStatus.SUPPORTED
    )

    assert result.citation_evaluation.citation_accuracy == 1.0
    assert result.citation_evaluation.citation_presence == 1.0
    assert result.citation_evaluation.citation_coverage == 1.0

    assert result.answer_completeness.score == 1.0

    assert 0.0 <= result.composite_confidence.score <= 1.0
    assert result.retrieval_confidence.score > 0.0


def test_end_to_end_invalid_citation():
    pipeline = make_pipeline(
        "Employees receive 24 days of annual leave [5]."
    )

    result = pipeline.run(
        "What is the annual leave allowance?"
    )

    assert result.citation_verification.valid == []
    assert result.citation_verification.invalid == [5]


def test_end_to_end_abstains_on_low_retrieval_confidence():
    pipeline = GroundedGenerationPipeline(
        retrieval_pipeline=FakeRetrievalPipeline([]),
        context_builder=ContextBuilder(),
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=FakeLLMProvider("Should not be used."),
        citation_extractor=CitationExtractor(),
        citation_verifier=CitationVerifier(),
        claim_extractor=ClaimExtractor(),
        claim_mapper=ClaimCitationMapper(),
        claim_verifier=FakeClaimVerifier(),
        citation_evaluator=CitationEvaluator(),
        completeness_evaluator=AnswerCompletenessEvaluator(),
        retrieval_confidence_calculator=(
            RetrievalConfidenceCalculator()
        ),
        composite_confidence_calculator=(
            CompositeConfidenceCalculator()
        ),
        abstention_policy=AbstentionPolicy(),
    )

    result = pipeline.run(
        "What is the annual leave allowance?"
    )

    assert result.abstention.should_abstain is True
    assert result.abstention.reason == "no_retrieval_context"
    assert result.answer == (
        "I don't have enough information in the provided "
        "documents to answer this question."
    )
