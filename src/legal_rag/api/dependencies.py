"""Lazy dependency construction for the API layer."""

from threading import Lock

from legal_rag.config import get_config
from legal_rag.monitoring import build_rag_tracer, get_metrics
from legal_rag.rag import (
    LegalRAGService,
    LegalRetriever,
    SentenceTransformerEmbedder,
    build_generator,
)
from legal_rag.storage import PostgresChunkRepository


class RAGConfigurationError(RuntimeError):
    """Raised when required RAG service configuration is unavailable."""


def _build_rag_service() -> LegalRAGService:
    config = get_config()
    repository = PostgresChunkRepository(config)
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    retriever = LegalRetriever(embedder, repository, config)
    try:
        generator = build_generator(config)
    except ValueError as exc:
        raise RAGConfigurationError("RAG generation service is not configured") from exc
    generator_model = (
        config.gemini_model
        if config.llm_provider.strip().lower() == "gemini"
        else config.vllm_model
    )
    return LegalRAGService(
        retriever,
        generator,
        metrics=get_metrics(),
        tracer=build_rag_tracer(config),
        provider=config.llm_provider,
        embedding_model=config.embedding_model,
        generator_model=generator_model,
        input_cost_per_million_tokens_usd=(
            config.llm_input_cost_per_million_tokens_usd
        ),
        output_cost_per_million_tokens_usd=(
            config.llm_output_cost_per_million_tokens_usd
        ),
    )


_service: LegalRAGService | None = None
_service_lock = Lock()


def get_rag_service() -> LegalRAGService:
    """Build one cached RAG service without duplicate concurrent cold starts."""

    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = _build_rag_service()
    return _service
