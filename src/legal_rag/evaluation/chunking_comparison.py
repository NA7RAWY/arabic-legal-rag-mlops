"""Run an isolated article-level versus long-article chunking experiment."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import psycopg

from legal_rag.config import AppConfig, get_config
from legal_rag.evaluation.dataset import (
    DEFAULT_EVALUATION_PATH,
    load_evaluation_dataset,
)
from legal_rag.evaluation.runner import RetrievalEvaluationReport, evaluate_retrieval
from legal_rag.ingestion import LegalChunk, chunk_articles, load_articles
from legal_rag.ingestion.experimental_chunker import (
    CorpusLengthAnalysis,
    SplitChunkConfig,
    analyze_corpus_lengths,
    split_long_articles,
)
from legal_rag.rag import LegalRetriever, SentenceTransformerEmbedder
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

EXPERIMENT_TABLE = "legal_chunks_chunking_experiment"
RUN_PURPOSE = "chunking-comparison"
TOP_K = 5


class _SemanticRetriever(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]: ...


@dataclass(frozen=True, slots=True)
class ChunkStructure:
    total_chunks: int
    articles_split: int
    mean_chunks_per_article: float
    max_chunks_per_article: int


@dataclass(frozen=True, slots=True)
class StrategyResult:
    strategy: str
    run_id: str
    structure: ChunkStructure
    retrieval: RetrievalEvaluationReport
    duplicate_candidates_removed: int


@dataclass(frozen=True, slots=True)
class ChunkingComparisonResult:
    corpus_analysis: CorpusLengthAnalysis
    split_config: SplitChunkConfig
    baseline: StrategyResult
    alternative: StrategyResult
    metric_winners: dict[str, str]
    index_size_increase: int
    index_size_increase_percentage: float


class ArticleDeduplicatingRetriever:
    """Return top-k distinct articles while retaining chunk similarity order."""

    def __init__(
        self, retriever: _SemanticRetriever, max_chunks_per_article: int
    ) -> None:
        self.retriever = retriever
        self.max_chunks_per_article = max_chunks_per_article
        self.duplicates_removed = 0

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if top_k is None or top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        candidates = self.retriever.search(
            query,
            top_k=top_k * self.max_chunks_per_article,
        )
        unique = []
        seen_articles: set[int] = set()
        for candidate in candidates:
            if candidate.article_number in seen_articles:
                self.duplicates_removed += 1
                continue
            seen_articles.add(candidate.article_number)
            unique.append(candidate)
            if len(unique) == top_k:
                break
        return unique


def chunk_structure(chunks: list[LegalChunk], article_count: int) -> ChunkStructure:
    if article_count <= 0:
        raise ValueError("article_count must be greater than zero")
    counts = Counter(chunk.article_number for chunk in chunks)
    return ChunkStructure(
        total_chunks=len(chunks),
        articles_split=sum(count > 1 for count in counts.values()),
        mean_chunks_per_article=len(chunks) / article_count,
        max_chunks_per_article=max(counts.values(), default=0),
    )


def metric_winners(
    baseline: RetrievalEvaluationReport,
    alternative: RetrievalEvaluationReport,
) -> dict[str, str]:
    """Name the winner for each metric, retaining ties explicitly."""

    fields = {
        "hit_rate": "hit_rate_at_k",
        "recall": "recall_at_k",
        "precision": "precision_at_k",
        "mrr": "mean_reciprocal_rank",
    }
    winners = {}
    for label, field in fields.items():
        baseline_value = getattr(baseline.aggregate_metrics, field)
        alternative_value = getattr(alternative.aggregate_metrics, field)
        if baseline_value == alternative_value:
            winners[label] = "tie"
        elif baseline_value > alternative_value:
            winners[label] = "one_article_per_chunk"
        else:
            winners[label] = "split_long_articles"
    return winners


def _recreate_experiment_table(config: AppConfig) -> None:
    with psycopg.connect(
        dbname=config.postgres_db,
        user=config.postgres_user,
        password=config.postgres_password,
        host=config.postgres_host,
        port=config.postgres_port,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {EXPERIMENT_TABLE}")
            cursor.execute(
                f"CREATE TABLE {EXPERIMENT_TABLE} (LIKE legal_chunks INCLUDING ALL)"
            )


def _log_strategy(
    *,
    tracker: MLflowTracker,
    experiment_config: RAGExperimentConfig,
    report: RetrievalEvaluationReport,
    structure: ChunkStructure,
    analysis: CorpusLengthAnalysis,
    split_config: SplitChunkConfig,
    comparison: dict[str, Any],
) -> str:
    metrics = report.aggregate_metrics
    with tracker.start_run(f"{RUN_PURPOSE}-{experiment_config.chunk_strategy}") as run:
        tracker.log_experiment_config(experiment_config)
        tracker.log_evaluation_metrics(
            {
                RAGEvaluationMetric.HIT_RATE_AT_K: metrics.hit_rate_at_k,
                RAGEvaluationMetric.MEAN_RECALL_AT_K: metrics.recall_at_k,
                RAGEvaluationMetric.MEAN_PRECISION_AT_K: metrics.precision_at_k,
                RAGEvaluationMetric.MRR: metrics.mean_reciprocal_rank,
                RAGEvaluationMetric.TOTAL_CHUNKS: structure.total_chunks,
                RAGEvaluationMetric.ARTICLES_SPLIT: structure.articles_split,
                RAGEvaluationMetric.MEAN_CHUNKS_PER_ARTICLE: (
                    structure.mean_chunks_per_article
                ),
                RAGEvaluationMetric.MAX_CHUNKS_PER_ARTICLE: (
                    structure.max_chunks_per_article
                ),
            }
        )
        tracker.log_config_bundle(asdict(experiment_config), "config/chunking.json")
        tracker.log_config_bundle(asdict(report), "evaluation/retrieval_report.json")
        tracker.log_config_bundle(asdict(analysis), "analysis/corpus_lengths.json")
        tracker.log_config_bundle(comparison, "evaluation/comparison_summary.json")
        return str(run.info.run_id)


def run_chunking_comparison(
    config: AppConfig | None = None,
    *,
    dataset_path: Path = DEFAULT_EVALUATION_PATH,
    split_config: SplitChunkConfig | None = None,
) -> ChunkingComparisonResult:
    """Index the isolated alternative and log two retrieval-only MLflow runs."""

    active_config = config or get_config()
    active_split_config = split_config or SplitChunkConfig()
    articles = load_articles(active_config.corpus_path)
    dataset = load_evaluation_dataset(
        dataset_path,
        corpus_path=active_config.corpus_path,
    )
    baseline_chunks = chunk_articles(articles)
    alternative_chunks = split_long_articles(articles, active_split_config)
    baseline_structure = chunk_structure(baseline_chunks, len(articles))
    alternative_structure = chunk_structure(alternative_chunks, len(articles))
    analysis = analyze_corpus_lengths(
        articles,
        split_threshold=active_split_config.split_threshold,
    )

    embedder = SentenceTransformerEmbedder(active_config.embedding_model)
    _recreate_experiment_table(active_config)
    experiment_repository = PostgresChunkRepository(
        active_config,
        table_name=EXPERIMENT_TABLE,
    )
    embeddings = embedder.embed_chunks(alternative_chunks)
    experiment_repository.upsert_chunks(alternative_chunks, embeddings)

    baseline_retriever = ArticleDeduplicatingRetriever(
        LegalRetriever(
            embedder,
            PostgresChunkRepository(active_config),
            active_config,
        ),
        baseline_structure.max_chunks_per_article,
    )
    alternative_retriever = ArticleDeduplicatingRetriever(
        LegalRetriever(embedder, experiment_repository, active_config),
        alternative_structure.max_chunks_per_article,
    )
    baseline_report = evaluate_retrieval(
        dataset,
        baseline_retriever,
        dataset_path=dataset_path,
        top_k=TOP_K,
    )
    alternative_report = evaluate_retrieval(
        dataset,
        alternative_retriever,
        dataset_path=dataset_path,
        top_k=TOP_K,
    )
    winners = metric_winners(baseline_report, alternative_report)
    index_increase = (
        alternative_structure.total_chunks - baseline_structure.total_chunks
    )
    comparison = {
        "metric_winners": winners,
        "index_size_increase": index_increase,
        "index_size_increase_percentage": (
            100.0 * index_increase / baseline_structure.total_chunks
        ),
        "baseline_duplicate_candidates_removed": baseline_retriever.duplicates_removed,
        "alternative_duplicate_candidates_removed": (
            alternative_retriever.duplicates_removed
        ),
    }

    commit = current_git_commit()
    dirty = current_git_dirty()
    corpus_version = corpus_sha256(active_config.corpus_path)
    dataset_sha256 = corpus_sha256(dataset_path)
    dataset_hash = f"sha256:{dataset_sha256}"
    base = build_baseline_config(
        active_config,
        git_commit=commit,
        git_dirty=dirty,
        corpus_version=corpus_version,
    )
    baseline_config = replace(
        base,
        retrieval_top_k=TOP_K,
        run_purpose=RUN_PURPOSE,
        eval_dataset=dataset.version,
        eval_dataset_hash=dataset_hash,
        eval_dataset_sha256=dataset_sha256,
        eval_cases=len(dataset.cases),
        split_threshold="not_applicable",
    )
    alternative_config = replace(
        baseline_config,
        chunk_strategy="split_long_articles",
        chunk_size=active_split_config.chunk_size,
        chunk_overlap=active_split_config.chunk_overlap,
        split_threshold=active_split_config.split_threshold,
    )
    tracker = MLflowTracker(active_config)
    baseline_run_id = _log_strategy(
        tracker=tracker,
        experiment_config=baseline_config,
        report=baseline_report,
        structure=baseline_structure,
        analysis=analysis,
        split_config=active_split_config,
        comparison=comparison,
    )
    alternative_run_id = _log_strategy(
        tracker=tracker,
        experiment_config=alternative_config,
        report=alternative_report,
        structure=alternative_structure,
        analysis=analysis,
        split_config=active_split_config,
        comparison=comparison,
    )
    return ChunkingComparisonResult(
        corpus_analysis=analysis,
        split_config=active_split_config,
        baseline=StrategyResult(
            "one_article_per_chunk",
            baseline_run_id,
            baseline_structure,
            baseline_report,
            baseline_retriever.duplicates_removed,
        ),
        alternative=StrategyResult(
            "split_long_articles",
            alternative_run_id,
            alternative_structure,
            alternative_report,
            alternative_retriever.duplicates_removed,
        ),
        metric_winners=winners,
        index_size_increase=index_increase,
        index_size_increase_percentage=comparison["index_size_increase_percentage"],
    )


def main() -> None:
    print(
        json.dumps(
            asdict(run_chunking_comparison()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
