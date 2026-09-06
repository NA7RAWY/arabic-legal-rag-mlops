"""FastAPI application for the Arabic Legal RAG service."""

import logging

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from legal_rag.api.dependencies import RAGConfigurationError, get_rag_service
from legal_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceResponse,
)
from legal_rag.config import AppConfig, get_config
from legal_rag.logging_conf import configure_logging
from legal_rag.rag import GenerationError, LegalRAGService, NoRetrievedContextError

logger = logging.getLogger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create the API application without initializing external services."""

    active_config = config if config is not None else get_config()
    configure_logging(active_config.log_level)
    application = FastAPI(
        title=active_config.app_name,
        version=active_config.app_version,
        description=f"Environment: {active_config.environment}",
    )

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

    @application.post("/ask", response_model=AskResponse)
    def ask(
        request: AskRequest,
        service: LegalRAGService = Depends(get_rag_service),
    ) -> AskResponse:
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

    return application


app = create_app()
