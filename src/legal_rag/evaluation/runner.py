"""Run deterministic retrieval evaluation and record it in MLflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from legal_rag.config import AppConfig, get_config
from legal_rag.evaluation.dataset import (
    DEFAULT_EVALUATION_PATH,
    EvaluationDataset,
    load_evaluation_dataset,
)
from legal_rag.evaluation.retrieval_metrics import (
    AggregateRetrievalMetrics,
    CaseRetrievalMetrics,
    aggregate_metrics,
    calculate_case_metrics,
)
from legal_rag.rag import LegalRetriever, SentenceTransformerEmbedder
from legal_rag.rag.generator import SYSTEM_INSTRUCTION
from legal_rag.storage import PostgresChunkRepository, RetrievalResult
from legal_rag.tracking.baseline import (
    build_baseline_config,
    corpus_sha256,
    current_git_commit,
    current_git_dirty,
)
from legal_rag.tracking.mlflow_tracker import (
    MLflowTracker,
    RAGEvaluationMetric,
    RAGExperimentConfig,
)

RUN_NAME = "module2-retrieval-eval"


class _Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


class _EvaluationTracker(Protocol):
    def start_run(
        self,
        run_name: str | None = None,
    ) -> AbstractContextManager[Any]: ...

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None: ...

    def log_evaluation_metrics(
        self,
        metrics: Mapping[RAGEvaluationMetric | str, float],
        *,
        step: int | None = None,
    ) -> None: ...

    def log_config_bundle(
        self,
        bundle: Mapping[str, Any],
        artifact_file: str = "config/rag_config.json",
    ) -> None: ...

    def log_prompt(
        self,
        prompt: str,
        artifact_file: str = "prompts/system_prompt.txt",
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RetrievedArticle:
    """Ranked article metadata retained in the evaluation report."""

    rank: int
    chunk_id: str
    article_number: int
    similarity: float


@dataclass(frozen=True, slots=True)
class CaseRetrievalResult:
    """Retrieved ranking and deterministic metrics for one case."""

    case_id: str
    question: str
    relevant_article_numbers: tuple[int, ...]
    retrieved_article_numbers: tuple[int, ...]
    retrieved: tuple[RetrievedArticle, ...]
    metrics: CaseRetrievalMetrics


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Complete deterministic report for one retrieval evaluation."""

    dataset_version: str
    dataset_path: str
    number_of_cases: int
    retrieval_top_k: int
    aggregate_metrics: AggregateRetrievalMetrics
    cases: tuple[CaseRetrievalResult, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRun:
    """MLflow identity paired with the evaluation report it records."""

    run_id: str
    report: RetrievalEvaluationReport


def evaluate_retrieval(
    dataset: EvaluationDataset,
    retriever: _Retriever,
    *,
    dataset_path: Path,
    top_k: int,
) -> RetrievalEvaluationReport:
    """Evaluate a retriever in dataset order without generation calls."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    case_results = []
    for evaluation_case in dataset.cases:
        search_results = retriever.search(evaluation_case.question, top_k=top_k)
        retrieved = tuple(
            RetrievedArticle(
                rank=rank,
                chunk_id=result.chunk_id,
                article_number=result.article_number,
                similarity=result.similarity,
            )
            for rank, result in enumerate(search_results, start=1)
        )
        retrieved_numbers = tuple(item.article_number for item in retrieved)
        metrics = calculate_case_metrics(
            retrieved_numbers,
            evaluation_case.relevant_article_numbers,
            top_k=top_k,
        )
        case_results.append(
            CaseRetrievalResult(
                case_id=evaluation_case.id,
                question=evaluation_case.question,
                relevant_article_numbers=evaluation_case.relevant_article_numbers,
                retrieved_article_numbers=retrieved_numbers,
                retrieved=retrieved,
                metrics=metrics,
            )
        )

    ordered_results = tuple(case_results)
    return RetrievalEvaluationReport(
        dataset_version=dataset.version,
        dataset_path=str(dataset_path),
        number_of_cases=len(ordered_results),
        retrieval_top_k=top_k,
        aggregate_metrics=aggregate_metrics(
            tuple(result.metrics for result in ordered_results)
        ),
        cases=ordered_results,
    )


def _build_retriever(config: AppConfig) -> LegalRetriever:
    repository = PostgresChunkRepository(config)
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    return LegalRetriever(embedder, repository, config)


def run_retrieval_evaluation(
    config: AppConfig | None = None,
    *,
    dataset_path: Path = DEFAULT_EVALUATION_PATH,
    retriever: _Retriever | None = None,
    tracker: _EvaluationTracker | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> RetrievalEvaluationRun:
    """Evaluate retrieval, log the report to MLflow, and return both IDs/data."""

    active_config = config if config is not None else get_config()
    dataset = load_evaluation_dataset(
        dataset_path,
        corpus_path=active_config.corpus_path,
    )
    active_retriever = retriever or _build_retriever(active_config)
    report = evaluate_retrieval(
        dataset,
        active_retriever,
        dataset_path=dataset_path,
        top_k=active_config.retrieval_top_k,
    )

    base_config = build_baseline_config(
        active_config,
        git_commit=git_commit if git_commit is not None else current_git_commit(),
        git_dirty=git_dirty if git_dirty is not None else current_git_dirty(),
        corpus_version=corpus_sha256(active_config.corpus_path),
    )
    experiment_config = replace(
        base_config,
        run_purpose="module2-retrieval-eval",
        eval_dataset=dataset.version,
    )
    active_tracker = tracker or MLflowTracker(active_config)
    metrics = report.aggregate_metrics

    with active_tracker.start_run(RUN_NAME) as run:
        active_tracker.log_experiment_config(experiment_config)
        active_tracker.log_evaluation_metrics(
            {
                RAGEvaluationMetric.RETRIEVAL_HIT_RATE_AT_K: metrics.hit_rate_at_k,
                RAGEvaluationMetric.RETRIEVAL_RECALL_AT_K: metrics.recall_at_k,
                RAGEvaluationMetric.RETRIEVAL_PRECISION_AT_K: metrics.precision_at_k,
                RAGEvaluationMetric.RETRIEVAL_MRR: metrics.mean_reciprocal_rank,
            }
        )
        active_tracker.log_config_bundle(
            asdict(report),
            "evaluation/retrieval_report.json",
        )
        active_tracker.log_config_bundle(
            asdict(experiment_config),
            "config/rag_config.json",
        )
        active_tracker.log_prompt(SYSTEM_INSTRUCTION)
        return RetrievalEvaluationRun(
            run_id=str(run.info.run_id),
            report=report,
        )


def main() -> None:
    """Run real local retrieval evaluation and print its summary as JSON."""

    outcome = run_retrieval_evaluation()
    print(
        json.dumps(
            {
                "run_id": outcome.run_id,
                "dataset_version": outcome.report.dataset_version,
                "number_of_cases": outcome.report.number_of_cases,
                "retrieval_top_k": outcome.report.retrieval_top_k,
                "aggregate_metrics": asdict(outcome.report.aggregate_metrics),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
