from app.embeddings.embedder import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return [
            [float(len(text))]
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        return [float(len(text))]


def test_embedding_provider_contract():

    provider = FakeEmbeddingProvider()

    documents = [
        "hello",
        "hello world",
    ]

    vectors = provider.embed_documents(documents)

    assert len(vectors) == 2
    assert vectors[0] == [5.0]
    assert vectors[1] == [11.0]


def test_query_embedding_contract():

    provider = FakeEmbeddingProvider()

    vector = provider.embed_query("hello")

    assert vector == [5.0]