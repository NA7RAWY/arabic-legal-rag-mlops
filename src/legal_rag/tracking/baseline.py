"""Configuration-only MLflow baseline for the Module 2 RAG system."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from legal_rag.config import AppConfig, get_config
from legal_rag.rag.generator import SYSTEM_INSTRUCTION
from legal_rag.tracking.mlflow_tracker import MLflowTracker, RAGExperimentConfig

PROMPT_VERSION = "grounded-system-v1"
RUN_NAME = "module2-rag-baseline"


class _BaselineTracker(Protocol):
    def start_run(
        self,
        run_name: str | None = None,
    ) -> AbstractContextManager[Any]: ...

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None: ...

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


def corpus_sha256(path: Path) -> str:
    """Return a deterministic SHA256 identifier for a corpus file."""

    digest = hashlib.sha256()
    with path.open("rb") as corpus_file:
        for block in iter(lambda: corpus_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit() -> str:
    """Return the repository commit that produced the baseline metadata."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_dirty() -> bool:
    """Return whether the repository has tracked or untracked changes."""

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_baseline_config(
    config: AppConfig,
    *,
    git_commit: str,
    git_dirty: bool,
    corpus_version: str,
) -> RAGExperimentConfig:
    """Build the real Module 2 baseline configuration."""

    return RAGExperimentConfig(
        chunk_strategy="one_article_per_chunk",
        chunk_size="article_level_variable_length",
        chunk_overlap=0,
        retrieval_top_k=config.retrieval_top_k,
        embedding_model=config.embedding_model,
        generator_model=config.gemini_model,
        prompt_version=PROMPT_VERSION,
        git_commit=git_commit,
        git_dirty=git_dirty,
        corpus_version=f"sha256:{corpus_version}",
        run_purpose="module2-baseline",
    )


def run_baseline_experiment(
    config: AppConfig | None = None,
    *,
    tracker: _BaselineTracker | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> str:
    """Log one configuration-only RAG baseline and return its MLflow run ID."""

    active_config = config if config is not None else get_config()
    active_tracker = tracker if tracker is not None else MLflowTracker(active_config)
    corpus_path = active_config.corpus_path
    experiment = build_baseline_config(
        active_config,
        git_commit=git_commit if git_commit is not None else current_git_commit(),
        git_dirty=git_dirty if git_dirty is not None else current_git_dirty(),
        corpus_version=corpus_sha256(corpus_path),
    )
    config_bundle = {
        **asdict(experiment),
        "corpus_path": str(corpus_path),
        "mlflow_experiment_name": active_config.mlflow_experiment_name,
    }

    with active_tracker.start_run(RUN_NAME) as run:
        active_tracker.log_experiment_config(experiment)
        active_tracker.log_config_bundle(config_bundle)
        active_tracker.log_prompt(SYSTEM_INSTRUCTION)
        return str(run.info.run_id)


def main() -> None:
    """Run the configuration-only baseline from the command line."""

    print(run_baseline_experiment())


if __name__ == "__main__":
    main()
