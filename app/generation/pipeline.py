from app.generation.context import ContextBuilder
from app.generation.llm import LLMProvider
from app.generation.prompt import GroundedPromptBuilder
from app.retrieval.pipeline import RetrievalPipeline


class GenerationPipeline:

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: GroundedPromptBuilder,
        llm_provider: LLMProvider,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider

    def generate(self, query: str) -> str:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        results = self.retrieval_pipeline.retrieve(
            query
        )

        context = self.context_builder.build(
            results
        )

        if not context.strip():
            raise ValueError(
                "No retrieval context available"
            )

        prompt = self.prompt_builder.build(
            query=query,
            context=context,
        )

        return self.llm_provider.generate(
            prompt
        )
