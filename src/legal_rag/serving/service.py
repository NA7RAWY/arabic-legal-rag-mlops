"""BentoML service that hosts the existing legal RAG ASGI application."""

import bentoml

from legal_rag.serving.app import create_serving_app

asgi_application = create_serving_app()


@bentoml.asgi_app(asgi_application, path="/")
@bentoml.service(
    name="arabic-legal-rag",
    traffic={"timeout": 300},
)
class LegalRAGBentoService:
    """BentoML lifecycle wrapper; FastAPI and LegalRAGService own request work."""
