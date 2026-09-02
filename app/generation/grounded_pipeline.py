from dataclasses import dataclass

from app.generation.abstention import AbstentionDecision, AbstentionPolicy
from app.generation.citation_evaluator import (
    CitationEvaluation,
    CitationEvaluator,
)
from app.generation.citation_verifier import (
    CitationVerificationResult,
    CitationVerifier,
)
from app.generation.citations import CitationExtractor
from app.generation.claim_mapping import (
    ClaimCitationMapper,
    ClaimEvidence,
)
from app.generation.claim_verifier import (
    ClaimVerification,
    ClaimVerifier,
)
from app.generation.claims import (
    AnswerClaim,
    ClaimExtractor,
)
from app.generation.completeness import (
    AnswerCompleteness,
    AnswerCompletenessEvaluator,
)
from app.generation.composite_confidence import (
    CompositeConfidence,
    CompositeConfidenceCalculator,
)
from app.generation.context import ContextBuilder
from app.generation.llm import LLMProvider
from app.generation.prompt import GroundedPromptBuilder
from app.generation.retrieval_confidence import (
    RetrievalConfidence,
    RetrievalConfidenceCalculator,
)
from app.retrieval.pipeline import RetrievalPipeline


@dataclass
class GroundedAnswer:
    query: str
    answer: str | None
    retrieval_confidence: RetrievalConfidence
    citation_verification: CitationVerificationResult
    claims: list[AnswerClaim]
    claim_evidence: list[ClaimEvidence]
    claim_verifications: list[ClaimVerification]
    citation_evaluation: CitationEvaluation
    answer_completeness: AnswerCompleteness
    composite_confidence: CompositeConfidence
    abstention: AbstentionDecision


class GroundedGenerationPipeline:

    ABSTENTION_MESSAGE = (
        "I don't have enough information in the provided "
        "documents to answer this question."
    )

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: GroundedPromptBuilder,
        llm_provider: LLMProvider,
        citation_extractor: CitationExtractor,
        citation_verifier: CitationVerifier,
        claim_extractor: ClaimExtractor,
        claim_mapper: ClaimCitationMapper,
        claim_verifier: ClaimVerifier,
        citation_evaluator: CitationEvaluator,
        completeness_evaluator: AnswerCompletenessEvaluator,
        retrieval_confidence_calculator: RetrievalConfidenceCalculator,
        composite_confidence_calculator: CompositeConfidenceCalculator,
        abstention_policy: AbstentionPolicy,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.citation_extractor = citation_extractor
        self.citation_verifier = citation_verifier
        self.claim_extractor = claim_extractor
        self.claim_mapper = claim_mapper
        self.claim_verifier = claim_verifier
        self.citation_evaluator = citation_evaluator
        self.completeness_evaluator = completeness_evaluator
        self.retrieval_confidence_calculator = (
            retrieval_confidence_calculator
        )
        self.composite_confidence_calculator = (
            composite_confidence_calculator
        )
        self.abstention_policy = abstention_policy

    def run(self, query: str) -> GroundedAnswer:

        if not query.strip():
            raise ValueError("query cannot be empty")

        results = self.retrieval_pipeline.retrieve(query)

        retrieval_confidence = (
            self.retrieval_confidence_calculator.calculate(results)
        )

        context = self.context_builder.build(results)

        if not context.strip():
            abstention = AbstentionDecision(
                should_abstain=True,
                reason="no_retrieval_context",
            )

            zero_citation_evaluation = self.citation_evaluator.evaluate([])
            zero_completeness = self.completeness_evaluator.evaluate(
                query,
                "",
            )
            composite = self.composite_confidence_calculator.calculate(
                retrieval_confidence=retrieval_confidence.score,
                citation_accuracy=0.0,
                citation_coverage=0.0,
                answer_completeness=zero_completeness.score,
            )

            return GroundedAnswer(
                query=query,
                answer=self.ABSTENTION_MESSAGE,
                retrieval_confidence=retrieval_confidence,
                citation_verification=CitationVerificationResult(
                    valid=[],
                    invalid=[],
                ),
                claims=[],
                claim_evidence=[],
                claim_verifications=[],
                citation_evaluation=zero_citation_evaluation,
                answer_completeness=zero_completeness,
                composite_confidence=composite,
                abstention=abstention,
            )

        prompt = self.prompt_builder.build(
            query=query,
            context=context,
        )

        answer = self.llm_provider.generate(prompt)

        citations = self.citation_extractor.extract(answer)

        citation_verification = self.citation_verifier.verify(
            citations,
            results,
        )

        claims = self.claim_extractor.extract(answer)

        claim_evidence = self.claim_mapper.map(
            claims,
            results,
        )

        claim_verifications = self.claim_verifier.verify(
            claim_evidence,
        )

        citation_evaluation = self.citation_evaluator.evaluate(
            claim_verifications,
        )

        answer_completeness = self.completeness_evaluator.evaluate(
            query,
            answer,
        )

        composite_confidence = (
            self.composite_confidence_calculator.calculate(
                retrieval_confidence=retrieval_confidence.score,
                citation_accuracy=citation_evaluation.citation_accuracy,
                citation_coverage=citation_evaluation.citation_coverage,
                answer_completeness=answer_completeness.score,
            )
        )

        abstention = self.abstention_policy.decide(
            composite_confidence=composite_confidence.score,
            retrieval_confidence=retrieval_confidence.score,
        )

        final_answer = (
            self.ABSTENTION_MESSAGE
            if abstention.should_abstain
            else answer
        )

        return GroundedAnswer(
            query=query,
            answer=final_answer,
            retrieval_confidence=retrieval_confidence,
            citation_verification=citation_verification,
            claims=claims,
            claim_evidence=claim_evidence,
            claim_verifications=claim_verifications,
            citation_evaluation=citation_evaluation,
            answer_completeness=answer_completeness,
            composite_confidence=composite_confidence,
            abstention=abstention,
        )
