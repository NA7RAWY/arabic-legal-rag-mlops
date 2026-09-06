"""Compare deterministic retrieval top-k configurations in MLflow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
from legal_rag.evaluation.retrieval_metrics import AggregateRetrievalMetrics
from legal_rag.evaluation.runner import (
    RetrievalEvaluationReport,
    _build_retriever,
    _Retriever,
    evaluate_retrieval,
)
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

DEFAULT_TOP_K_VALUES = (3, 5, 8)
RUN_PURPOSE = "retrieval-top-k-comparison"


class _Tracker(Protocol):
    def start_run(self, run_name: str | None = None) -> AbstractContextManager[Any]: ...

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


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    top_k: int
    run_id: str
    metrics: AggregateRetrievalMetrics


@dataclass(frozen=True, slots=True)
class RetrievalComparisonSummary:
    top_k_values: tuple[int, ...]
    best_hit_rate_top_k: int
    best_recall_top_k: int
    best_precision_top_k: int
    best_mrr_top_k: int
    recommendation: str


@dataclass(frozen=True, slots=True)
class RetrievalComparisonResult:
    experiment_name: str
    corpus_version: str
    eval_dataset_version: str
    eval_dataset_hash: str
    runs: tuple[ComparisonRun, ...]
    summary: RetrievalComparisonSummary


def _best_top_k(
    reports: Sequence[RetrievalEvaluationReport],
    metric: str,
) -> int:
    """Choose the smallest top-k when metric values tie."""

    return max(
        reports,
        key=lambda report: (
            getattr(report.aggregate_metrics, metric),
            -report.retrieval_top_k,
        ),
    ).retrieval_top_k


def summarize_comparison(
    reports: Sequence[RetrievalEvaluationReport],
) -> RetrievalComparisonSummary:
    """Summarize metric leaders and the precision/recall tradeoff."""

    if not reports:
        raise ValueError("At least one retrieval report is required")
    ordered = tuple(sorted(reports, key=lambda report: report.retrieval_top_k))
    top_k_values = tuple(report.retrieval_top_k for report in ordered)
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("Retrieval top_k values must be unique")
    return RetrievalComparisonSummary(
        top_k_values=top_k_values,
        best_hit_rate_top_k=_best_top_k(ordered, "hit_rate_at_k"),
        best_recall_top_k=_best_top_k(ordered, "recall_at_k"),
        best_precision_top_k=_best_top_k(ordered, "precision_at_k"),
        best_mrr_top_k=_best_top_k(ordered, "mean_reciprocal_rank"),
        recommendation=(
            "Select top_k from measured quality and context-budget needs: smaller "
            "values favor precision and lower context cost; larger values can improve "
            "hit rate and recall. The production default is unchanged."
        ),
    )


def _evaluate_configurations(
    dataset: EvaluationDataset,
    retriever: _Retriever,
    dataset_path: Path,
    top_k_values: Sequence[int],
) -> tuple[RetrievalEvaluationReport, ...]:
    if not top_k_values or any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k values must be positive and non-empty")
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("top_k values must be unique")
    return tuple(
        evaluate_retrieval(
            dataset,
            retriever,
            dataset_path=dataset_path,
            top_k=top_k,
        )
        for top_k in top_k_values
    )


def run_retrieval_comparison(
    config: AppConfig | None = None,
    *,
    dataset_path: Path = DEFAULT_EVALUATION_PATH,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    retriever: _Retriever | None = None,
    tracker: _Tracker | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> RetrievalComparisonResult:
    """Evaluate and log one MLflow run per top-k configuration."""

    active_config = config or get_config()
    dataset = load_evaluation_dataset(
        dataset_path,
        corpus_path=active_config.corpus_path,
    )
    active_retriever = retriever or _build_retriever(active_config)
    reports = _evaluate_configurations(
        dataset,
        active_retriever,
        dataset_path,
        top_k_values,
    )
    summary = summarize_comparison(reports)
    commit = git_commit if git_commit is not None else current_git_commit()
    dirty = git_dirty if git_dirty is not None else current_git_dirty()
    corpus_version = f"sha256:{corpus_sha256(active_config.corpus_path)}"
    dataset_sha256 = corpus_sha256(dataset_path)
    dataset_hash = f"sha256:{dataset_sha256}"
    tracker_instance = tracker or MLflowTracker(active_config)
    base_config = build_baseline_config(
        active_config,
        git_commit=commit,
        git_dirty=dirty,
        corpus_version=corpus_version.removeprefix("sha256:"),
    )

    logged_runs = []
    for report in reports:
        experiment_config = replace(
            base_config,
            retrieval_top_k=report.retrieval_top_k,
            run_purpose=RUN_PURPOSE,
            eval_dataset=dataset.version,
            eval_dataset_hash=dataset_hash,
            eval_dataset_sha256=dataset_sha256,
            eval_cases=len(dataset.cases),
        )
        metrics = report.aggregate_metrics
        with tracker_instance.start_run(
            f"{RUN_PURPOSE}-k{report.retrieval_top_k}"
        ) as run:
            tracker_instance.log_experiment_config(experiment_config)
            tracker_instance.log_evaluation_metrics(
                {
                    RAGEvaluationMetric.HIT_RATE_AT_K: metrics.hit_rate_at_k,
                    RAGEvaluationMetric.MEAN_RECALL_AT_K: metrics.recall_at_k,
                    RAGEvaluationMetric.MEAN_PRECISION_AT_K: metrics.precision_at_k,
                    RAGEvaluationMetric.MRR: metrics.mean_reciprocal_rank,
                }
            )
            tracker_instance.log_config_bundle(
                asdict(report),
                "evaluation/retrieval_report.json",
            )
            tracker_instance.log_config_bundle(
                asdict(experiment_config),
                "config/retrieval_config.json",
            )
            tracker_instance.log_config_bundle(
                asdict(summary),
                "evaluation/comparison_summary.json",
            )
            logged_runs.append(
                ComparisonRun(
                    top_k=report.retrieval_top_k,
                    run_id=str(run.info.run_id),
                    metrics=metrics,
                )
            )

    return RetrievalComparisonResult(
        experiment_name=active_config.mlflow_experiment_name,
        corpus_version=corpus_version,
        eval_dataset_version=dataset.version,
        eval_dataset_hash=dataset_hash,
        runs=tuple(logged_runs),
        summary=summary,
    )


def main() -> None:
    result = run_retrieval_comparison()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
