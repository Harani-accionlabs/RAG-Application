import logging
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedding(Embeddings):
    """LangChain-compatible wrapper around a SentenceTransformer model."""

    def __init__(self, model: str) -> None:
        logger.info(f"Loading embedding model: {model}")
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch-encode instead of one-by-one — much faster
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()