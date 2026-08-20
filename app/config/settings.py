from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    openai_api_key: str | None = None

    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"

    chroma_persist_dir: str = "./storage/chroma"
    bm25_persist_dir: str = "./storage/bm25"

    collection_name: str = "hybrid_rag_documents"

    chunk_size: int = 800
    chunk_overlap: int = 120

    dense_top_k: int = 10
    sparse_top_k: int = 10
    fusion_top_k: int = 20
    final_top_k: int = 5

    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    min_retrieval_score: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()