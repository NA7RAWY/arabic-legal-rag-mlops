"""Load and validate corpus-grounded legal RAG evaluation cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.ingestion import LegalArticle, load_articles

DEFAULT_EVALUATION_PATH = Path("data/evaluation/legal_rag_eval_v1.json")
_DATASET_FIELDS = {"version", "cases"}
_CASE_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "relevant_article_numbers",
    "category",
    "language",
    "source_rationale",
    "supporting_excerpts",
}
_EXCERPT_FIELDS = {"article_number", "excerpt"}


@dataclass(frozen=True, slots=True)
class SupportingExcerpt:
    """An exact corpus excerpt supporting one evaluation case."""

    article_number: int
    excerpt: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One manually curated, corpus-grounded evaluation example."""

    id: str
    question: str
    reference_answer: str
    relevant_article_numbers: tuple[int, ...]
    category: str
    language: str
    source_rationale: str
    supporting_excerpts: tuple[SupportingExcerpt, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A versioned and deterministically ordered evaluation dataset."""

    version: str
    cases: tuple[EvaluationCase, ...]


def _require_exact_fields(
    record: dict[str, Any],
    expected: set[str],
    location: str,
) -> None:
    missing = expected - record.keys()
    unexpected = record.keys() - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Malformed {location}: {'; '.join(details)}")


def _non_empty_string(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty string")
    return value.strip()


def _parse_excerpt(value: Any, case_id: str, index: int) -> SupportingExcerpt:
    location = f"case {case_id} supporting_excerpts[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    _require_exact_fields(value, _EXCERPT_FIELDS, location)
    article_number = value["article_number"]
    if isinstance(article_number, bool) or not isinstance(article_number, int):
        raise ValueError(f"{location}.article_number must be an integer")
    return SupportingExcerpt(
        article_number=article_number,
        excerpt=_non_empty_string(value["excerpt"], "excerpt", location),
    )


def _parse_case(value: Any, index: int) -> EvaluationCase:
    location = f"cases[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    _require_exact_fields(value, _CASE_FIELDS, location)
    case_id = _non_empty_string(value["id"], "id", location)

    article_numbers = value["relevant_article_numbers"]
    if not isinstance(article_numbers, list) or not article_numbers:
        raise ValueError(
            f"case {case_id}.relevant_article_numbers must be a non-empty list"
        )
    if any(
        isinstance(number, bool) or not isinstance(number, int)
        for number in article_numbers
    ):
        raise ValueError(
            f"case {case_id}.relevant_article_numbers must contain integers"
        )
    if len(set(article_numbers)) != len(article_numbers):
        raise ValueError(f"case {case_id}.relevant_article_numbers contains duplicates")

    excerpts = value["supporting_excerpts"]
    if not isinstance(excerpts, list) or not excerpts:
        raise ValueError(f"case {case_id}.supporting_excerpts must be a non-empty list")

    language = _non_empty_string(value["language"], "language", location)
    if language not in {"ar", "en"}:
        raise ValueError(f"case {case_id}.language must be 'ar' or 'en'")

    return EvaluationCase(
        id=case_id,
        question=_non_empty_string(value["question"], "question", location),
        reference_answer=_non_empty_string(
            value["reference_answer"], "reference_answer", location
        ),
        relevant_article_numbers=tuple(article_numbers),
        category=_non_empty_string(value["category"], "category", location),
        language=language,
        source_rationale=_non_empty_string(
            value["source_rationale"], "source_rationale", location
        ),
        supporting_excerpts=tuple(
            _parse_excerpt(excerpt, case_id, excerpt_index)
            for excerpt_index, excerpt in enumerate(excerpts)
        ),
    )


def _validate_grounding(
    evaluation_case: EvaluationCase,
    articles: dict[int, LegalArticle],
) -> None:
    excerpt_numbers = tuple(
        excerpt.article_number for excerpt in evaluation_case.supporting_excerpts
    )
    if excerpt_numbers != evaluation_case.relevant_article_numbers:
        raise ValueError(
            f"case {evaluation_case.id} supporting excerpt articles must exactly "
            "match relevant_article_numbers in order"
        )

    for supporting_excerpt in evaluation_case.supporting_excerpts:
        article = articles.get(supporting_excerpt.article_number)
        if article is None:
            raise ValueError(
                f"case {evaluation_case.id} references missing article "
                f"{supporting_excerpt.article_number}"
            )
        if article.text_ar is None or not article.text_ar.strip():
            raise ValueError(
                f"case {evaluation_case.id} references article "
                f"{supporting_excerpt.article_number} without Arabic text"
            )
        if supporting_excerpt.excerpt not in article.text_ar:
            raise ValueError(
                f"case {evaluation_case.id} excerpt is not present in article "
                f"{supporting_excerpt.article_number} Arabic text"
            )


def load_evaluation_dataset(
    path: Path = DEFAULT_EVALUATION_PATH,
    *,
    corpus_path: Path,
) -> EvaluationDataset:
    """Load a versioned evaluation set and validate it against the corpus."""

    try:
        with path.open("r", encoding="utf-8") as dataset_file:
            raw = json.load(dataset_file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid evaluation dataset JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Evaluation dataset root must be an object")
    _require_exact_fields(raw, _DATASET_FIELDS, "evaluation dataset")
    version = _non_empty_string(raw["version"], "version", "evaluation dataset")
    raw_cases = raw["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("evaluation dataset.cases must be a list")
    if not raw_cases:
        raise ValueError("evaluation dataset.cases must not be empty")

    cases = tuple(_parse_case(case, index) for index, case in enumerate(raw_cases))
    case_ids = [case.id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Evaluation dataset contains duplicate case IDs")
    normalized_questions = [case.question.casefold() for case in cases]
    if len(set(normalized_questions)) != len(normalized_questions):
        raise ValueError("Evaluation dataset contains duplicate questions")

    articles = {
        article.article_number: article for article in load_articles(corpus_path)
    }
    for evaluation_case in cases:
        _validate_grounding(evaluation_case, articles)
    return EvaluationDataset(version=version, cases=cases)
