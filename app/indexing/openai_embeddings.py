from openai import OpenAI

from app.config.settings import get_settings
from app.indexing.embeddings import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(self):
        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

        self.model = settings.embedding_model

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

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

        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )

        return response.data[0].embedding