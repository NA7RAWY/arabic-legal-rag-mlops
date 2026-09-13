"""Framework-neutral construction for the BentoML-mounted ASGI application."""

from fastapi import FastAPI

from legal_rag.api.app import create_app
from legal_rag.config import AppConfig


def create_serving_app(config: AppConfig | None = None) -> FastAPI:
    """Return the existing FastAPI contract for mounting by BentoML.

    External RAG dependencies remain lazy because ``create_app`` resolves
    ``get_rag_service`` only when the ask endpoint is called.
    """

    return create_app(config)
