"""End-to-end generation and RAGAS evaluation with MLflow tracking."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
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
    generated_answer: str
    retrieval_metrics: CaseRetrievalMetrics
    ragas_metrics: RagasScores


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
    retrieval_top_k: int
    number_of_cases: int
    aggregate_retrieval_metrics: AggregateRetrievalMetrics
    aggregate_ragas_metrics: RagasScores
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
) -> EndToEndReport:
    """Retrieve, generate once, and judge each case in dataset order."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    case_results: list[EndToEndCaseResult] = []
    for case in dataset.cases:
        retrieved = retriever.search(case.question, top_k=top_k)
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
        answer = generator.generate(case.question, build_legal_context(retrieved))
        context_texts = [context.text for context in contexts]
        ragas_scores = evaluator.evaluate(
            question=case.question,
            answer=answer,
            contexts=context_texts,
            reference_answer=case.reference_answer,
        )
        retrieved_numbers = tuple(context.article_number for context in contexts)
        case_results.append(
            EndToEndCaseResult(
                case_id=case.id,
                question=case.question,
                reference_answer=case.reference_answer,
                relevant_article_numbers=case.relevant_article_numbers,
                retrieved_article_numbers=retrieved_numbers,
                retrieved_contexts=contexts,
                generated_answer=answer,
                retrieval_metrics=calculate_case_metrics(
                    retrieved_numbers,
                    case.relevant_article_numbers,
                    top_k=top_k,
                ),
                ragas_metrics=ragas_scores,
            )
        )
    ordered = tuple(case_results)
    if not ordered:
        raise ValueError("Evaluation dataset must contain at least one case")
    return EndToEndReport(
        dataset_version=dataset.version,
        dataset_path=str(dataset_path),
        created_at=datetime.now(UTC).isoformat(),
        corpus_version=corpus_version,
        git_commit=git_commit,
        git_dirty=git_dirty,
        generation_model=generation_model,
        evaluation_model=evaluation_model,
        retrieval_top_k=top_k,
        number_of_cases=len(ordered),
        aggregate_retrieval_metrics=aggregate_metrics(
            tuple(case.retrieval_metrics for case in ordered)
        ),
        aggregate_ragas_metrics=_mean_ragas(
            tuple(case.ragas_metrics for case in ordered)
        ),
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
    )
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
    )
    active_tracker = tracker or MLflowTracker(active_config)
    retrieval = report.aggregate_retrieval_metrics
    ragas = report.aggregate_ragas_metrics
    metrics = {
        RAGEvaluationMetric.RETRIEVAL_HIT_RATE_AT_K: retrieval.hit_rate_at_k,
        RAGEvaluationMetric.RETRIEVAL_RECALL_AT_K: retrieval.recall_at_k,
        RAGEvaluationMetric.RETRIEVAL_PRECISION_AT_K: retrieval.precision_at_k,
        RAGEvaluationMetric.RETRIEVAL_MRR: retrieval.mean_reciprocal_rank,
        RAGEvaluationMetric.RAGAS_FAITHFULNESS: ragas.faithfulness,
        RAGEvaluationMetric.RAGAS_ANSWER_RELEVANCY: ragas.answer_relevancy,
        RAGEvaluationMetric.RAGAS_CONTEXT_RECALL: ragas.context_recall,
        RAGEvaluationMetric.RAGAS_CONTEXT_PRECISION: ragas.context_precision,
    }
    with active_tracker.start_run(RUN_NAME) as run:
        active_tracker.log_experiment_config(experiment)
        active_tracker.log_evaluation_metrics(metrics)
        active_tracker.log_config_bundle(
            asdict(report), "evaluation/end_to_end_report.json"
        )
        active_tracker.log_config_bundle(asdict(experiment), "config/rag_config.json")
        active_tracker.log_config_bundle(
            {
                "ragas_version": "0.4.3",
                "metrics_api": "ragas.metrics.collections",
                "evaluation_model": active_config.evaluation_model,
            },
            "config/evaluation_config.json",
        )
        active_tracker.log_prompt(SYSTEM_INSTRUCTION)
        return EndToEndEvaluationRun(str(run.info.run_id), report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    outcome = run_end_to_end_evaluation(
        case_limit=args.limit,
        log_to_mlflow=not args.no_mlflow,
    )
    print(
        json.dumps(
            {"run_id": outcome.run_id, **asdict(outcome.report)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
