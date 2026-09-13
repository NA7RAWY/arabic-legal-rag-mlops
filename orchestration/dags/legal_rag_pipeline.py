"""Manual Airflow orchestration for legal RAG maintenance and evaluation.

Airflow coordinates existing project entrypoints; application modules retain all
corpus, indexing, retrieval, generation, and evaluation business logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

VALIDATE_EVALUATION_DATASET_COMMAND = "python -m legal_rag.evaluation.summary"
RUN_RETRIEVAL_EVALUATION_COMMAND = "python -m legal_rag.evaluation.runner"

# Live generation and judge calls are deliberately opt-in to protect provider quota.
RUN_RAG_EVALUATION_COMMAND = """if [ "${ENABLE_LIVE_RAG_EVALUATION:-false}" = "true" ]; then python -m legal_rag.evaluation.end_to_end; else echo 'Live RAG evaluation skipped; set ENABLE_LIVE_RAG_EVALUATION=true explicitly to run it'; fi"""

# This file lives at <project>/orchestration/dags/legal_rag_pipeline.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ARGS = {
    "owner": "legal-rag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def resolve_project_path(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a configured path without changing the worker process directory."""

    return path if path.is_absolute() else (project_root / path).resolve()


def validate_corpus_callable(project_root: str) -> int:
    """Validate canonical corpus cardinality and contiguous article numbering."""

    from legal_rag.config import get_config
    from legal_rag.ingestion import load_articles

    config = get_config()
    corpus_path = resolve_project_path(config.corpus_path, Path(project_root))
    articles = load_articles(corpus_path)
    article_numbers = [article.article_number for article in articles]
    if len(articles) != 1149:
        raise ValueError(f"Expected 1149 articles, found {len(articles)}")
    if article_numbers != list(range(1, 1150)):
        raise ValueError("Article numbers must be contiguous from 1 through 1149")
    print("Validated 1149 canonical articles with contiguous numbering")
    return len(articles)


def rebuild_vector_index_callable(project_root: str) -> int:
    """Rebuild the vector index through the existing ingestion pipeline."""

    from dataclasses import replace

    from legal_rag.config import get_config
    from legal_rag.ingestion import index_corpus
    from legal_rag.rag import SentenceTransformerEmbedder
    from legal_rag.storage import PostgresChunkRepository

    config = get_config()
    config = replace(
        config,
        corpus_path=resolve_project_path(config.corpus_path, Path(project_root)),
    )
    indexed_count = index_corpus(
        PostgresChunkRepository(config),
        SentenceTransformerEmbedder(config.embedding_model),
        config,
    )
    print(f"Indexed {indexed_count} legal chunks")
    return indexed_count


with DAG(
    dag_id="legal_rag_pipeline",
    description="Validate data, rebuild the vector index, and evaluate legal RAG",
    default_args=DEFAULT_ARGS,
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    tags=["legal-rag", "module-3"],
) as dag:
    validate_corpus = PythonOperator(
        task_id="validate_corpus",
        python_callable=validate_corpus_callable,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
    )

    validate_evaluation_dataset = BashOperator(
        task_id="validate_evaluation_dataset",
        bash_command=VALIDATE_EVALUATION_DATASET_COMMAND,
        cwd=str(PROJECT_ROOT),
    )

    rebuild_vector_index = PythonOperator(
        task_id="rebuild_vector_index",
        python_callable=rebuild_vector_index_callable,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
    )

    run_retrieval_evaluation = BashOperator(
        task_id="run_retrieval_evaluation",
        bash_command=RUN_RETRIEVAL_EVALUATION_COMMAND,
        cwd=str(PROJECT_ROOT),
    )

    run_rag_evaluation = BashOperator(
        task_id="run_rag_evaluation",
        bash_command=RUN_RAG_EVALUATION_COMMAND,
        cwd=str(PROJECT_ROOT),
    )

    (
        validate_corpus
        >> validate_evaluation_dataset
        >> rebuild_vector_index
        >> run_retrieval_evaluation
        >> run_rag_evaluation
    )
