"""FastAPI application for the Arabic Legal RAG service."""

import json
import logging
from collections.abc import Iterator

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from legal_rag.api.dependencies import RAGConfigurationError, get_rag_service
from legal_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceResponse,
)
from legal_rag.config import AppConfig, get_config
from legal_rag.logging_conf import configure_logging
from legal_rag.monitoring import PrometheusMetrics, PrometheusMiddleware, get_metrics
from legal_rag.rag import (
    GenerationError,
    LegalRAGService,
    LegalRAGStreamResult,
    NoRetrievedContextError,
)

logger = logging.getLogger(__name__)


def _sse_event(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_events(result: LegalRAGStreamResult) -> Iterator[str]:
    chunks = iter(result.chunks)
    try:
        first_chunk = next(chunks)
    except (GenerationError, StopIteration):
        logger.error("Streaming ask request failed before first provider chunk")
        yield _sse_event("error", {"detail": "Answer generation provider failed"})
        return

    sources = [
        SourceResponse(
            chunk_id=source.chunk_id,
            article_number=source.article_number,
            citation=source.citation,
            language=source.language,
            similarity=source.similarity,
        ).model_dump()
        for source in result.retrieved_sources
    ]
    yield _sse_event(
        "sources",
        {"question": result.question, "sources": sources},
    )
    yield _sse_event("token", {"text": first_chunk})
    try:
        for chunk in chunks:
            yield _sse_event("token", {"text": chunk})
    except GenerationError:
        logger.error("Streaming ask request failed: generation provider error")
        yield _sse_event("error", {"detail": "Answer generation provider failed"})
        return
    yield _sse_event("done", {})


def create_app(
    config: AppConfig | None = None,
    metrics: PrometheusMetrics | None = None,
) -> FastAPI:
    """Create the API application without initializing external services."""

    active_config = config if config is not None else get_config()
    active_metrics = metrics if metrics is not None else get_metrics()
    configure_logging(active_config.log_level)
    application = FastAPI(
        title=active_config.app_name,
        version=active_config.app_version,
        description=f"Environment: {active_config.environment}",
    )
    application.add_middleware(PrometheusMiddleware, metrics=active_metrics)

    @application.exception_handler(RAGConfigurationError)
    async def handle_configuration_error(
        request: Request,
        exc: RAGConfigurationError,
    ) -> JSONResponse:
        logger.error("Ask request failed: RAG service is not configured")
        return JSONResponse(
            status_code=503,
            content={"detail": "RAG service is not configured"},
        )

    @application.exception_handler(GenerationError)
    async def handle_generation_error(
        request: Request,
        exc: GenerationError,
    ) -> JSONResponse:
        logger.error("Ask request failed: generation provider error")
        return JSONResponse(
            status_code=502,
            content={"detail": "Answer generation provider failed"},
        )

    @application.exception_handler(NoRetrievedContextError)
    async def handle_missing_context(
        request: Request,
        exc: NoRetrievedContextError,
    ) -> JSONResponse:
        logger.warning("Ask request completed without retrieved legal context")
        return JSONResponse(
            status_code=404,
            content={"detail": "No relevant legal context was found"},
        )

    @application.exception_handler(psycopg.Error)
    async def handle_database_error(
        request: Request,
        exc: psycopg.Error,
    ) -> JSONResponse:
        logger.error("Ask request failed: retrieval database unavailable")
        return JSONResponse(
            status_code=503,
            content={"detail": "Legal retrieval service is unavailable"},
        )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app=active_config.app_name,
            version=active_config.app_version,
        )

    @application.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(active_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @application.post("/ask", response_model=AskResponse)
    def ask(
        request: AskRequest,
        service: LegalRAGService = Depends(get_rag_service),
    ) -> AskResponse:
        active_metrics.record_ask("full")
        top_k = (
            active_config.retrieval_top_k if request.top_k is None else request.top_k
        )
        logger.info(
            "Received ask request with question length %d and top_k %d",
            len(request.question),
            top_k,
        )
        result = service.answer(request.question, top_k=top_k)
        response = AskResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                SourceResponse(
                    chunk_id=source.chunk_id,
                    article_number=source.article_number,
                    citation=source.citation,
                    language=source.language,
                    similarity=source.similarity,
                )
                for source in result.retrieved_sources
            ],
        )
        logger.info(
            "Completed ask request successfully with %d sources",
            len(response.sources),
        )
        return response

    @application.post("/ask/stream")
    def ask_stream(
        request: AskRequest,
        service: LegalRAGService = Depends(get_rag_service),
    ) -> StreamingResponse:
        active_metrics.record_ask("stream")
        top_k = (
            active_config.retrieval_top_k if request.top_k is None else request.top_k
        )
        logger.info(
            "Received streaming ask request with question length %d and top_k %d",
            len(request.question),
            top_k,
        )
        result = service.stream_answer(request.question, top_k=top_k)
        return StreamingResponse(
            _stream_events(result),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return application


app = create_app()
