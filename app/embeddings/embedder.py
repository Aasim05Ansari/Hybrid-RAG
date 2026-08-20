from openai import OpenAI

from app.config.settings import settings
from app.indexing.embeddings import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model: str | None = None,
    ):
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI embeddings"
            )

        self.model = model or settings.embedding_model

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        if not text.strip():
            raise ValueError(
                "Query text cannot be empty"
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding