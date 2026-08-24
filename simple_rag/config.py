from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator, field_validator


class Settings(BaseSettings):
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer model name")
    chunk_size: int = Field(default=800, description="Text splitter chunk size")
    chunk_overlap: int = Field(default=150, description="Text splitter chunk overlap")
    persist_directory: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data",
        description="Chroma persistence directory"
    )
    top_k: int = Field(default=4, description="Number of results to return per query")
    log_level: str = Field(default="INFO", description="Logging level")

    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model for reranking")
    rerank_candidate_pool: int = Field(default=20, description="Number of candidates to retrieve before reranking")

    llm_model: str = Field(
    default="openai/gpt-oss-20b",
    description="Groq-hosted model for generation"
)
    groq_api_key: str | None = Field(default=None, description="Groq API key")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    @model_validator(mode='after')
    def validate_groq_api_key(self):
        if not self.groq_api_key:
            raise ValueError("Groq API key is required for answer generation")
        return self

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return upper_v


settings = Settings()