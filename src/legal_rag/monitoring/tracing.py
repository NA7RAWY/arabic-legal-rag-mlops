"""Fail-open Langfuse v4 tracing for legal RAG executions."""

import logging
from typing import Any, Protocol

from legal_rag.config import AppConfig
from legal_rag.storage import RetrievalResult

logger = logging.getLogger(__name__)


class RAGObservation(Protocol):
    """A privacy-conscious operation within one RAG trace."""

    def succeed(self, metadata: dict[str, Any] | None = None) -> None: ...

    def fail(self, status: str) -> None: ...

    def end(self) -> None: ...


class RAGTrace(Protocol):
    """One logical RAG request and its child observations."""

    def start_retrieval(self) -> RAGObservation: ...

    def start_generation(self) -> RAGObservation: ...

    def succeed(self) -> None: ...

    def fail(self, status: str) -> None: ...

    def end(self) -> None: ...


class RAGTracer(Protocol):
    """Factory for request-scoped traces."""

    def start_request(
        self,
        *,
        mode: str,
        top_k: int | None,
        provider: str,
        embedding_model: str,
        generator_model: str,
    ) -> RAGTrace: ...


class NoOpObservation:
    """Observation implementation used when tracing is disabled or unavailable."""

    def succeed(self, metadata: dict[str, Any] | None = None) -> None:
        pass

    def fail(self, status: str) -> None:
        pass

    def end(self) -> None:
        pass


class NoOpTrace(NoOpObservation):
    """Trace implementation that deliberately records nothing."""

    def start_retrieval(self) -> RAGObservation:
        return NoOpObservation()

    def start_generation(self) -> RAGObservation:
        return NoOpObservation()


class NoOpTracer:
    """Safe default tracer for local development and tests."""

    def start_request(
        self,
        *,
        mode: str,
        top_k: int | None,
        provider: str,
        embedding_model: str,
        generator_model: str,
    ) -> RAGTrace:
        return NoOpTrace()


class _SafeLangfuseObservation:
    """Isolate all SDK recording failures from application work."""

    def __init__(
        self,
        observation: Any | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._observation = observation
        self._metadata = metadata or {}
        self._ended = False

    def succeed(self, metadata: dict[str, Any] | None = None) -> None:
        self._update(
            metadata={**self._metadata, "status": "success", **(metadata or {})}
        )

    def fail(self, status: str) -> None:
        self._update(
            metadata={**self._metadata, "status": "error"},
            level="ERROR",
            status_message=status,
        )

    def _update(self, **kwargs: Any) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(**kwargs)
        except Exception:
            logger.warning("Langfuse observation update failed")

    def end(self) -> None:
        if self._observation is None or self._ended:
            return
        self._ended = True
        try:
            self._observation.end()
        except Exception:
            logger.warning("Langfuse observation finalization failed")


class _SafeLangfuseTrace(_SafeLangfuseObservation):
    def __init__(
        self,
        observation: Any | None,
        *,
        provider: str,
        mode: str,
        top_k: int | None,
        generator_model: str,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__(observation, metadata)
        self._provider = provider
        self._mode = mode
        self._top_k = top_k
        self._generator_model = generator_model

    def _start_child(self, **kwargs: Any) -> RAGObservation:
        if self._observation is None:
            return NoOpObservation()
        try:
            child = self._observation.start_observation(**kwargs)
        except Exception:
            logger.warning("Langfuse child observation creation failed")
            return NoOpObservation()
        return _SafeLangfuseObservation(child, kwargs.get("metadata"))

    def start_retrieval(self) -> RAGObservation:
        return self._start_child(
            name="retrieval",
            as_type="retriever",
            metadata={"mode": self._mode, "requested_top_k": self._top_k},
        )

    def start_generation(self) -> RAGObservation:
        return self._start_child(
            name="generation",
            as_type="generation",
            model=self._generator_model,
            metadata={"provider": self._provider, "mode": self._mode},
        )


class LangfuseTracer:
    """Create privacy-conscious Langfuse v4 observations through one client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def start_request(
        self,
        *,
        mode: str,
        top_k: int | None,
        provider: str,
        embedding_model: str,
        generator_model: str,
    ) -> RAGTrace:
        metadata = {
            "endpoint": "/ask" if mode == "full" else "/ask/stream",
            "mode": mode,
            "top_k": top_k,
            "provider": provider,
            "embedding_model": embedding_model,
            "content_capture": "disabled",
        }
        try:
            root = self._client.start_observation(
                name="legal-rag-request",
                as_type="chain",
                metadata=metadata,
            )
        except Exception:
            logger.warning("Langfuse request observation creation failed")
            return NoOpTrace()
        return _SafeLangfuseTrace(
            root,
            provider=provider,
            mode=mode,
            top_k=top_k,
            generator_model=generator_model,
            metadata=metadata,
        )


def build_rag_tracer(
    config: AppConfig,
    *,
    client_factory: Any | None = None,
) -> RAGTracer:
    """Build a v4 client-backed tracer, or a no-op tracer on any setup failure."""

    if not config.langfuse_enabled:
        return NoOpTracer()
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning("Langfuse tracing disabled because credentials are incomplete")
        return NoOpTracer()
    if not config.langfuse_base_url.strip():
        logger.warning("Langfuse tracing disabled because its base URL is empty")
        return NoOpTracer()

    try:
        if client_factory is None:
            from langfuse import get_client

            client_factory = get_client
        # Langfuse v4 get_client() initializes the default single-project client.
        # Passing public_key here would only look up an already initialized keyed
        # client and would return a disabled client when none exists yet.
        client = client_factory()
        logger.info("Langfuse tracing is enabled and its client is initialized")
        return LangfuseTracer(client)
    except Exception:
        logger.warning("Langfuse client initialization failed; tracing is disabled")
        return NoOpTracer()


def retrieval_metadata(sources: list[RetrievalResult]) -> dict[str, Any]:
    """Return bounded identifiers only, never retrieved legal text."""

    return {
        "source_count": len(sources),
        "article_numbers": [source.article_number for source in sources],
        "chunk_ids": [source.chunk_id for source in sources],
    }
