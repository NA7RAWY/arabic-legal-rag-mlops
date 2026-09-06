"""MLflow tracking abstraction for RAG experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

from legal_rag.config import AppConfig, get_config


class RAGEvaluationMetric(StrEnum):
    """Reserved names for future measured RAG evaluation metrics."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    RETRIEVAL_HIT_RATE_AT_K = "retrieval_hit_rate_at_k"
    RETRIEVAL_RECALL_AT_K = "retrieval_recall_at_k"
    RETRIEVAL_PRECISION_AT_K = "retrieval_precision_at_k"
    RETRIEVAL_MRR = "retrieval_mrr"
    HIT_RATE_AT_K = "hit_rate_at_k"
    MEAN_RECALL_AT_K = "mean_recall_at_k"
    MEAN_PRECISION_AT_K = "mean_precision_at_k"
    MRR = "mrr"
    TOTAL_CHUNKS = "total_chunks"
    ARTICLES_SPLIT = "articles_split"
    MEAN_CHUNKS_PER_ARTICLE = "mean_chunks_per_article"
    MAX_CHUNKS_PER_ARTICLE = "max_chunks_per_article"
    RAGAS_FAITHFULNESS = "ragas_faithfulness"
    RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
    RAGAS_CONTEXT_RECALL = "ragas_context_recall"
    RAGAS_CONTEXT_PRECISION = "ragas_context_precision"


@dataclass(frozen=True, slots=True)
class RAGExperimentConfig:
    """Parameters and provenance recorded for one RAG experiment."""

    chunk_strategy: str
    chunk_size: int | str
    chunk_overlap: int
    retrieval_top_k: int
    embedding_model: str
    generator_model: str
    prompt_version: str
    git_commit: str
    corpus_version: str
    framework: str = "custom-rag"
    project_type: str = "arabic-legal-rag"
    run_purpose: str | None = None
    git_dirty: bool = False
    eval_dataset: str | None = None
    evaluator_model: str | None = None
    eval_dataset_hash: str | None = None
    eval_dataset_sha256: str | None = None
    eval_cases: int | None = None
    split_threshold: int | str | None = None


class _MLflowModule(Protocol):
    def set_tracking_uri(self, uri: str) -> None: ...

    def set_experiment(self, name: str) -> Any: ...

    def start_run(
        self, *, run_name: str | None = None
    ) -> AbstractContextManager[Any]: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def set_tags(self, tags: Mapping[str, str]) -> None: ...

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None: ...

    def log_dict(self, dictionary: Mapping[str, Any], artifact_file: str) -> None: ...

    def log_text(self, text: str, artifact_file: str) -> None: ...


class MLflowTracker:
    """Configure MLflow and log stable RAG experiment metadata."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        mlflow_module: _MLflowModule | None = None,
    ) -> None:
        self.config = config if config is not None else get_config()
        self._mlflow = mlflow_module

    def _client(self) -> _MLflowModule:
        if self._mlflow is None:
            import mlflow

            self._mlflow = mlflow
        return self._mlflow

    def configure(self) -> None:
        """Select the configured tracking server and experiment."""

        client = self._client()
        client.set_tracking_uri(self.config.mlflow_tracking_uri)
        client.set_experiment(self.config.mlflow_experiment_name)

    def start_run(
        self,
        run_name: str | None = None,
    ) -> AbstractContextManager[Any]:
        """Configure tracking and return an MLflow run context manager."""

        self.configure()
        return self._client().start_run(run_name=run_name)

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None:
        """Log RAG parameters and provenance tags to the active run."""

        values = asdict(experiment)
        params = {
            key: values[key]
            for key in (
                "chunk_strategy",
                "chunk_size",
                "chunk_overlap",
                "retrieval_top_k",
                "embedding_model",
                "generator_model",
                "prompt_version",
                "corpus_version",
            )
        }
        if experiment.eval_dataset is not None:
            params["eval_dataset"] = experiment.eval_dataset
        if experiment.eval_dataset_hash is not None:
            params["eval_dataset_hash"] = experiment.eval_dataset_hash
        if experiment.eval_dataset_sha256 is not None:
            params["eval_dataset_sha256"] = experiment.eval_dataset_sha256
        if experiment.eval_cases is not None:
            params["eval_cases"] = experiment.eval_cases
        if experiment.evaluator_model is not None:
            params["evaluator_model"] = experiment.evaluator_model
        if experiment.split_threshold is not None:
            params["split_threshold"] = experiment.split_threshold
        tags = {
            "git_commit": experiment.git_commit,
            "corpus_version": experiment.corpus_version,
            "framework": experiment.framework,
            "project_type": experiment.project_type,
            "git_dirty": str(experiment.git_dirty).lower(),
        }
        if experiment.run_purpose is not None:
            tags["run_purpose"] = experiment.run_purpose
        client = self._client()
        client.log_params(params)
        client.set_tags(tags)

    def log_evaluation_metrics(
        self,
        metrics: Mapping[RAGEvaluationMetric | str, float],
        *,
        step: int | None = None,
    ) -> None:
        """Log measured RAG metrics without supplying or inventing values."""

        allowed = {metric.value for metric in RAGEvaluationMetric}
        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            metric_name = str(name)
            if metric_name not in allowed:
                raise ValueError(f"Unsupported RAG evaluation metric: {metric_name}")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Metric {metric_name} must be finite")
            normalized[metric_name] = numeric_value
        self._client().log_metrics(normalized, step=step)

    def log_config_bundle(
        self,
        bundle: Mapping[str, Any],
        artifact_file: str = "config/rag_config.json",
    ) -> None:
        """Log a JSON-serializable RAG configuration bundle."""

        self._client().log_dict(bundle, artifact_file)

    def log_prompt(
        self,
        prompt: str,
        artifact_file: str = "prompts/system_prompt.txt",
    ) -> None:
        """Log prompt text as a run artifact."""

        if not prompt.strip():
            raise ValueError("Prompt text must not be empty")
        self._client().log_text(prompt, artifact_file)
