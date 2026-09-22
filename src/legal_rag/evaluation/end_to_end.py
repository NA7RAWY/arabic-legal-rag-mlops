"""End-to-end generation and RAGAS evaluation with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from legal_rag.config import AppConfig, get_config
from legal_rag.evaluation.dataset import (
    DEFAULT_EVALUATION_PATH,
    EvaluationDataset,
    load_evaluation_dataset,
)
from legal_rag.evaluation.ragas_evaluator import (
    GeminiRagasEvaluator,
    RagasEvaluationError,
    RagasEvaluator,
    RagasScores,
)
from legal_rag.evaluation.retrieval_metrics import (
    AggregateRetrievalMetrics,
    CaseRetrievalMetrics,
    aggregate_metrics,
    calculate_case_metrics,
)
from legal_rag.rag import LegalRetriever, SentenceTransformerEmbedder
from legal_rag.rag.generator import (
    SYSTEM_INSTRUCTION,
    GeminiGenerator,
    LLMGenerator,
    build_legal_context,
)
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

RUN_NAME = "module2-end-to-end-eval"
DEFAULT_REPORT_PATH = Path("artifacts/evaluation/ragas_50_case_report.json")
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)

logger = logging.getLogger(__name__)


class _Retriever(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]: ...


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
        self, bundle: Mapping[str, Any], artifact_file: str = "config/rag_config.json"
    ) -> None: ...
    def log_prompt(
        self, prompt: str, artifact_file: str = "prompts/system_prompt.txt"
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluatedContext:
    rank: int
    chunk_id: str
    article_number: int
    text: str
    language: str
    citation: str
    similarity: float


@dataclass(frozen=True, slots=True)
class EndToEndCaseResult:
    case_id: str
    question: str
    reference_answer: str
    relevant_article_numbers: tuple[int, ...]
    retrieved_article_numbers: tuple[int, ...]
    retrieved_contexts: tuple[EvaluatedContext, ...]
    generated_answer: str | None
    retrieval_metrics: CaseRetrievalMetrics | None
    ragas_metrics: RagasScores | None
    status: str
    failure: CaseEvaluationFailure | None


@dataclass(frozen=True, slots=True)
class CaseEvaluationFailure:
    """Sanitized, machine-readable failure for one evaluation case."""

    stage: str
    error_type: str
    message: str
    provider_status_code: int | None


@dataclass(frozen=True, slots=True)
class EndToEndReport:
    dataset_version: str
    dataset_path: str
    created_at: str
    corpus_version: str
    git_commit: str
    git_dirty: bool
    generation_model: str
    evaluation_model: str
    ragas_version: str
    ragas_metric_names: tuple[str, ...]
    retrieval_top_k: int
    case_delay_seconds: float
    number_of_cases: int
    successful_cases: int
    failed_cases: int
    aggregate_retrieval_metrics: AggregateRetrievalMetrics | None
    aggregate_ragas_metrics: RagasScores | None
    cases: tuple[EndToEndCaseResult, ...]


@dataclass(frozen=True, slots=True)
class EndToEndEvaluationRun:
    run_id: str
    report: EndToEndReport


def _mean_ragas(scores: tuple[RagasScores, ...]) -> RagasScores:
    if not scores:
        raise ValueError("At least one RAGAS score is required")
    return RagasScores(
        faithfulness=fmean(score.faithfulness for score in scores),
        answer_relevancy=fmean(score.answer_relevancy for score in scores),
        context_recall=fmean(score.context_recall for score in scores),
        context_precision=fmean(score.context_precision for score in scores),
    )


def _installed_ragas_version() -> str:
    try:
        return version("ragas")
    except PackageNotFoundError:
        return "unavailable"


def _provider_status_code(exc: Exception) -> int | None:
    """Find an integer provider status without exposing exception text."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for value in (
            getattr(current, "code", None),
            getattr(current, "status_code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ):
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        current = current.__cause__ or current.__context__
    return None


def _failure(stage: str, exc: Exception) -> CaseEvaluationFailure:
    return CaseEvaluationFailure(
        stage=stage,
        error_type=type(exc).__name__,
        message=f"{stage} failed",
        provider_status_code=_provider_status_code(exc),
    )


def _write_report(report: EndToEndReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_end_to_end(
    dataset: EvaluationDataset,
    *,
    retriever: _Retriever,
    generator: LLMGenerator,
    evaluator: RagasEvaluator,
    dataset_path: Path,
    top_k: int,
    corpus_version: str,
    git_commit: str,
    git_dirty: bool,
    generation_model: str,
    evaluation_model: str,
    case_delay_seconds: float = 0,
    sleep_callable: Callable[[float], None] = time.sleep,
) -> EndToEndReport:
    """Retrieve, generate once, and judge each case in dataset order."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if case_delay_seconds < 0:
        raise ValueError("case_delay_seconds must not be negative")
    case_results: list[EndToEndCaseResult] = []
    for case_index, case in enumerate(dataset.cases):
        retrieved: list[RetrievalResult] = []
        contexts: tuple[EvaluatedContext, ...] = ()
        retrieved_numbers: tuple[int, ...] = ()
        retrieval_metrics: CaseRetrievalMetrics | None = None
        answer: str | None = None
        ragas_scores: RagasScores | None = None
        failure: CaseEvaluationFailure | None = None
        try:
            retrieved = retriever.search(case.question, top_k=top_k)
            if not retrieved:
                raise ValueError("retrieval returned no contexts")
        except Exception as exc:
            failure = _failure("retrieval", exc)

        contexts = tuple(
            EvaluatedContext(
                rank=rank,
                chunk_id=result.chunk_id,
                article_number=result.article_number,
                text=result.text,
                language=result.language,
                citation=result.citation,
                similarity=result.similarity,
            )
            for rank, result in enumerate(retrieved, start=1)
        )
        retrieved_numbers = tuple(context.article_number for context in contexts)
        if retrieved:
            retrieval_metrics = calculate_case_metrics(
                retrieved_numbers,
                case.relevant_article_numbers,
                top_k=top_k,
            )
        if failure is None:
            try:
                answer = generator.generate(
                    case.question, build_legal_context(retrieved)
                )
            except Exception as exc:
                failure = _failure("generation", exc)
        if failure is None and answer is not None:
            try:
                evaluated_scores = evaluator.evaluate(
                    question=case.question,
                    answer=answer,
                    contexts=[context.text for context in contexts],
                    reference_answer=case.reference_answer,
                )
                if not isinstance(evaluated_scores, RagasScores):
                    raise RagasEvaluationError(
                        "RAGAS evaluator returned an invalid score bundle"
                    )
                ragas_scores = evaluated_scores
            except Exception as exc:
                failure = _failure("ragas", exc)

        case_results.append(
            EndToEndCaseResult(
                case_id=case.id,
                question=case.question,
                reference_answer=case.reference_answer,
                relevant_article_numbers=case.relevant_article_numbers,
                retrieved_article_numbers=retrieved_numbers,
                retrieved_contexts=contexts,
                generated_answer=answer,
                retrieval_metrics=retrieval_metrics,
                ragas_metrics=ragas_scores,
                status="success" if failure is None else "failed",
                failure=failure,
            )
        )
        if failure is None:
            logger.info("Completed evaluation case %s", case.id)
        else:
            logger.warning(
                "Evaluation case %s failed during %s with %s (provider status %s)",
                case.id,
                failure.stage,
                failure.error_type,
                failure.provider_status_code or "unavailable",
            )
        if case_delay_seconds and case_index < len(dataset.cases) - 1:
            sleep_callable(case_delay_seconds)

    ordered = tuple(case_results)
    if not ordered:
        raise ValueError("Evaluation dataset must contain at least one case")
    valid_retrieval = tuple(
        case.retrieval_metrics for case in ordered if case.retrieval_metrics is not None
    )
    valid_ragas = tuple(
        case.ragas_metrics for case in ordered if case.ragas_metrics is not None
    )
    successful_cases = len(valid_ragas)
    return EndToEndReport(
        dataset_version=dataset.version,
        dataset_path=str(dataset_path),
        created_at=datetime.now(UTC).isoformat(),
        corpus_version=corpus_version,
        git_commit=git_commit,
        git_dirty=git_dirty,
        generation_model=generation_model,
        evaluation_model=evaluation_model,
        ragas_version=_installed_ragas_version(),
        ragas_metric_names=RAGAS_METRIC_NAMES,
        retrieval_top_k=top_k,
        case_delay_seconds=case_delay_seconds,
        number_of_cases=len(ordered),
        successful_cases=successful_cases,
        failed_cases=len(ordered) - successful_cases,
        aggregate_retrieval_metrics=(
            aggregate_metrics(valid_retrieval) if valid_retrieval else None
        ),
        aggregate_ragas_metrics=_mean_ragas(valid_ragas) if valid_ragas else None,
        cases=ordered,
    )


def _build_live_components(
    config: AppConfig,
) -> tuple[_Retriever, LLMGenerator, RagasEvaluator]:
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    retriever = LegalRetriever(embedder, PostgresChunkRepository(config=config), config)
    generator = GeminiGenerator(config)
    evaluator = GeminiRagasEvaluator(
        api_key=config.gemini_api_key or "",
        model_name=config.evaluation_model,
        embedder=embedder,
    )
    return retriever, generator, evaluator


def run_end_to_end_evaluation(
    config: AppConfig | None = None,
    *,
    dataset_path: Path = DEFAULT_EVALUATION_PATH,
    case_limit: int | None = None,
    retriever: _Retriever | None = None,
    generator: LLMGenerator | None = None,
    evaluator: RagasEvaluator | None = None,
    tracker: _Tracker | None = None,
    log_to_mlflow: bool = True,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    output_path: Path | None = None,
    case_delay_seconds: float = 0,
) -> EndToEndEvaluationRun:
    """Run an end-to-end benchmark and optionally persist one MLflow run."""

    active_config = config or get_config()
    if case_limit is not None and case_limit <= 0:
        raise ValueError("case_limit must be greater than zero")
    dataset = load_evaluation_dataset(
        dataset_path, corpus_path=active_config.corpus_path
    )
    if case_limit is not None:
        dataset = EvaluationDataset(dataset.version, dataset.cases[:case_limit])
    if retriever is None or generator is None or evaluator is None:
        live_retriever, live_generator, live_evaluator = _build_live_components(
            active_config
        )
        retriever = retriever or live_retriever
        generator = generator or live_generator
        evaluator = evaluator or live_evaluator
    commit = git_commit if git_commit is not None else current_git_commit()
    dirty = git_dirty if git_dirty is not None else current_git_dirty()
    corpus_version = f"sha256:{corpus_sha256(active_config.corpus_path)}"
    report = evaluate_end_to_end(
        dataset,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        dataset_path=dataset_path,
        top_k=active_config.retrieval_top_k,
        corpus_version=corpus_version,
        git_commit=commit,
        git_dirty=dirty,
        generation_model=active_config.gemini_model,
        evaluation_model=active_config.evaluation_model,
        case_delay_seconds=case_delay_seconds,
    )
    if output_path is not None:
        _write_report(report, output_path)
    if not log_to_mlflow:
        return EndToEndEvaluationRun(run_id="", report=report)

    experiment = replace(
        build_baseline_config(
            active_config,
            git_commit=commit,
            git_dirty=dirty,
            corpus_version=corpus_version.removeprefix("sha256:"),
        ),
        run_purpose=RUN_NAME,
        eval_dataset=dataset.version,
        evaluator_model=active_config.evaluation_model,
        eval_dataset_sha256=corpus_sha256(dataset_path),
        eval_cases=len(dataset.cases),
    )
    active_tracker = tracker or MLflowTracker(active_config)
    metrics: dict[RAGEvaluationMetric, float] = {
        RAGEvaluationMetric.EVALUATION_SUCCESSFUL_CASES: float(report.successful_cases),
        RAGEvaluationMetric.EVALUATION_FAILED_CASES: float(report.failed_cases),
    }
    retrieval = report.aggregate_retrieval_metrics
    if retrieval is not None:
        metrics.update(
            {
                RAGEvaluationMetric.RETRIEVAL_HIT_RATE_AT_K: retrieval.hit_rate_at_k,
                RAGEvaluationMetric.RETRIEVAL_RECALL_AT_K: retrieval.recall_at_k,
                RAGEvaluationMetric.RETRIEVAL_PRECISION_AT_K: retrieval.precision_at_k,
                RAGEvaluationMetric.RETRIEVAL_MRR: retrieval.mean_reciprocal_rank,
            }
        )
    ragas = report.aggregate_ragas_metrics
    if ragas is not None:
        metrics.update(
            {
                RAGEvaluationMetric.RAGAS_FAITHFULNESS: ragas.faithfulness,
                RAGEvaluationMetric.RAGAS_ANSWER_RELEVANCY: ragas.answer_relevancy,
                RAGEvaluationMetric.RAGAS_CONTEXT_RECALL: ragas.context_recall,
                RAGEvaluationMetric.RAGAS_CONTEXT_PRECISION: ragas.context_precision,
            }
        )
    with active_tracker.start_run(RUN_NAME) as run:
        active_tracker.log_experiment_config(experiment)
        active_tracker.log_evaluation_metrics(metrics)
        active_tracker.log_config_bundle(
            asdict(report), "evaluation/end_to_end_report.json"
        )
        active_tracker.log_config_bundle(asdict(experiment), "config/rag_config.json")
        active_tracker.log_config_bundle(
            {
                "ragas_version": report.ragas_version,
                "metrics_api": "ragas.metrics.collections",
                "evaluation_model": active_config.evaluation_model,
                "case_delay_seconds": case_delay_seconds,
            },
            "config/evaluation_config.json",
        )
        active_tracker.log_prompt(SYSTEM_INSTRUCTION)
        return EndToEndEvaluationRun(str(run.info.run_id), report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--case-delay-seconds", type=float, default=0)
    args = parser.parse_args()
    outcome = run_end_to_end_evaluation(
        case_limit=args.limit,
        log_to_mlflow=not args.no_mlflow,
        output_path=args.output,
        case_delay_seconds=args.case_delay_seconds,
    )
    print(
        json.dumps(
            {"run_id": outcome.run_id, **asdict(outcome.report)},
            ensure_ascii=False,
            indent=2,
        )
    )
    if outcome.report.failed_cases:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
