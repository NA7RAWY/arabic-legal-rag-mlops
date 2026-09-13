"""Application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _postgres_port() -> int:
    value = os.getenv("POSTGRES_PORT", "5432")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("POSTGRES_PORT must be an integer") from exc


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Settings shared across the application."""

    app_name: str = "Arabic Legal RAG"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )
    )
    retrieval_top_k: int = 5
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini")
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    )
    evaluation_model: str = field(
        default_factory=lambda: os.getenv("EVALUATION_MODEL", "gemini-3.7-flash")
    )
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    vllm_model: str = field(
        default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    )
    vllm_api_key: str = field(
        default_factory=lambda: os.getenv("VLLM_API_KEY", "EMPTY")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "arabic-legal-rag-dev"
        )
    )
    mlflow_backend_store_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_BACKEND_STORE_URI", "sqlite:////mlflow/mlflow.db"
        )
    )
    mlflow_artifact_root: str = field(
        default_factory=lambda: os.getenv("MLFLOW_ARTIFACT_ROOT", "/mlflow/artifacts")
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "legal_rag")
    )
    postgres_user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "legal_rag")
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "legal_rag_dev")
    )
    postgres_host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost")
    )
    postgres_port: int = field(default_factory=_postgres_port)
    corpus_path: Path = Path("data/processed/civil_code_articles_clean_v2.json")


def get_config() -> AppConfig:
    """Return the application configuration."""

    return AppConfig()
