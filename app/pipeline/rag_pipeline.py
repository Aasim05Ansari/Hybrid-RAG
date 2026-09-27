from app.config.settings import settings

from app.embeddings.embedder import OpenAIEmbeddingProvider

from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import ChromaVectorStore

from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import LexicalReranker
from app.retrieval.rrf import RRFFusion
from app.retrieval.sparse import SparseRetriever

from app.generation.abstention import AbstentionPolicy
from app.generation.citation_evaluator import CitationEvaluator
from app.generation.citation_verifier import CitationVerifier
from app.generation.citations import CitationExtractor
from app.generation.claim_mapping import ClaimCitationMapper
from app.generation.claims import ClaimExtractor
from app.generation.claim_verifier import ClaimVerifier
from app.generation.completeness import AnswerCompletenessEvaluator
from app.generation.composite_confidence import (
    CompositeConfidenceCalculator,
)
from app.generation.context import ContextBuilder
from app.generation.grounded_pipeline import (
    GroundedAnswer,
    GroundedGenerationPipeline,
)
from app.generation.llm import OpenAILLMProvider
from app.generation.lexical_claim_verifier import (
    LexicalClaimVerifier,
)
from app.generation.prompt import GroundedPromptBuilder
from app.generation.retrieval_confidence import (
    RetrievalConfidenceCalculator,
)


class RAGPipeline:
    """
    Composition root for the Hybrid RAG system.

    This class is responsible for wiring together:
        retrieval components
        generation components
        grounding components
        confidence components
        abstention policy

    It does not implement retrieval or generation logic itself.
    """

    def __init__(
        self,
        grounded_generation_pipeline: GroundedGenerationPipeline,
    ):
        if grounded_generation_pipeline is None:
            raise ValueError(
                "grounded_generation_pipeline is required"
            )

        self.grounded_generation_pipeline = (
            grounded_generation_pipeline
        )

    def run(self, query: str) -> GroundedAnswer:
        """
        Run the complete grounded RAG pipeline.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        return self.grounded_generation_pipeline.run(
            query
        )

    @classmethod
    def create(
        cls,
        bm25_store: BM25Store,
        embedding_provider=None,
        vector_store=None,
        llm_provider=None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> "RAGPipeline":
        """
        Construct the complete production RAG pipeline.

        BM25Store is injected because the current BM25 implementation
        is in-memory and therefore cannot reconstruct its index after
        process restart.
        """

        if bm25_store is None:
            raise ValueError(
                "bm25_store is required"
            )

        # ---------------------------------------------------------
        # 1. Providers
        # ---------------------------------------------------------

        embedding_provider = (
            embedding_provider
            or OpenAIEmbeddingProvider()
        )

        llm_provider = (
            llm_provider
            or OpenAILLMProvider()
        )

        # ---------------------------------------------------------
        # 2. Stores
        # ---------------------------------------------------------

        vector_store = (
            vector_store
            or ChromaVectorStore(
                persist_directory=settings.chroma_persist_dir,
                collection_name=settings.collection_name,
            )
        )

        # ---------------------------------------------------------
        # 3. Retrieval
        # ---------------------------------------------------------

        dense_retriever = DenseRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        sparse_retriever = SparseRetriever(
            bm25_store=bm25_store,
        )

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=RRFFusion(),
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            fusion_top_k=settings.fusion_top_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        )

        reranker = LexicalReranker()

        retrieval_pipeline = RetrievalPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            final_top_k=settings.final_top_k,
        )

        # ---------------------------------------------------------
        # 4. Generation / grounding components
        # ---------------------------------------------------------

        context_builder = ContextBuilder()

        prompt_builder = GroundedPromptBuilder()

        citation_extractor = CitationExtractor()

        citation_verifier = CitationVerifier()

        claim_extractor = ClaimExtractor()

        claim_mapper = ClaimCitationMapper()

        # Default deterministic verifier.
        #
        # This keeps RAGPipeline.create() usable without making
        # an additional LLM call for every claim.
        claim_verifier = (
            claim_verifier
            or LexicalClaimVerifier()
        )

        citation_evaluator = CitationEvaluator()

        completeness_evaluator = (
            AnswerCompletenessEvaluator()
        )

        retrieval_confidence_calculator = (
            RetrievalConfidenceCalculator()
        )

        composite_confidence_calculator = (
            CompositeConfidenceCalculator()
        )

        abstention_policy = AbstentionPolicy()

        # ---------------------------------------------------------
        # 5. Grounded generation pipeline
        # ---------------------------------------------------------

        grounded_generation_pipeline = (
            GroundedGenerationPipeline(
                retrieval_pipeline=retrieval_pipeline,
                context_builder=context_builder,
                prompt_builder=prompt_builder,
                llm_provider=llm_provider,
                citation_extractor=citation_extractor,
                citation_verifier=citation_verifier,
                claim_extractor=claim_extractor,
                claim_mapper=claim_mapper,
                claim_verifier=claim_verifier,
                citation_evaluator=citation_evaluator,
                completeness_evaluator=completeness_evaluator,
                retrieval_confidence_calculator=(
                    retrieval_confidence_calculator
                ),
                composite_confidence_calculator=(
                    composite_confidence_calculator
                ),
                abstention_policy=abstention_policy,
            )
        )

        return cls(
            grounded_generation_pipeline=(
                grounded_generation_pipeline
            )
        )