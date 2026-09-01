"""Sentence-transformer embedding support for legal text."""

from typing import Any, Protocol, cast

from legal_rag.config import get_config
from legal_rag.ingestion import LegalChunk


class _SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool,
    ) -> Any: ...


class SentenceTransformerEmbedder:
    """Create normalized multilingual E5 embeddings on demand."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_config().embedding_model
        self._model: _SentenceEncoder | None = None

    def _load_model(self) -> _SentenceEncoder:
        from sentence_transformers import SentenceTransformer

        return cast(_SentenceEncoder, SentenceTransformer(self.model_name))

    def _get_model(self) -> _SentenceEncoder:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @staticmethod
    def _to_list(encoded: Any) -> Any:
        return encoded.tolist() if hasattr(encoded, "tolist") else encoded

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with the E5 passage prefix in input order."""

        if not texts:
            return []

        prefixed_texts = [f"passage: {text}" for text in texts]
        encoded = self._get_model().encode(
            prefixed_texts,
            normalize_embeddings=True,
        )
        vectors = self._to_list(encoded)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed one non-empty query with the E5 query prefix."""

        if not text.strip():
            raise ValueError("Query text must not be empty or whitespace-only")

        encoded = self._get_model().encode(
            f"query: {text}",
            normalize_embeddings=True,
        )
        vector = self._to_list(encoded)
        return [float(value) for value in vector]

    def embed_chunks(self, chunks: list[LegalChunk]) -> list[list[float]]:
        """Embed legal chunk text while preserving chunk order."""

        return self.embed_documents([chunk.text for chunk in chunks])
