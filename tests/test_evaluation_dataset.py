"""Tests for evaluation dataset loading and corpus grounding."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from legal_rag.evaluation import EvaluationCase, load_evaluation_dataset

CANONICAL_CORPUS = Path("data/processed/civil_code_articles_clean_v2.json")
EVALUATION_DATASET = Path("data/evaluation/legal_rag_eval_v1.json")


def article_record(article_number: int, text_ar: str) -> dict[str, Any]:
    return {
        "article_number": article_number,
        "book": None,
        "chapter": None,
        "section": None,
        "topic": None,
        "text_ar": text_ar,
        "text_en": "English fallback",
        "is_repealed": False,
        "source_page": 1,
        "citation": f"Article {article_number}",
    }


def case_record(case_id: str, article_number: int, excerpt: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "question": "ما القاعدة القانونية؟",
        "reference_answer": "هذه هي القاعدة القانونية المقررة في النص.",
        "relevant_article_numbers": [article_number],
        "category": "اختبار",
        "language": "ar",
        "source_rationale": "النص المشار إليه يقرر القاعدة مباشرة.",
        "supporting_excerpts": [{"article_number": article_number, "excerpt": excerpt}],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def temporary_files(
    tmp_path: Path,
    cases: list[dict[str, Any]],
) -> tuple[Path, Path]:
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "evaluation.json"
    article_numbers = {
        number
        for case in cases
        for number in case.get("relevant_article_numbers", [])
        if isinstance(number, int) and not isinstance(number, bool)
    }
    write_json(
        corpus_path,
        [
            article_record(number, f"نص المادة {number} الداعم")
            for number in article_numbers
        ],
    )
    write_json(dataset_path, {"version": "test-v1", "cases": cases})
    return dataset_path, corpus_path


@pytest.mark.skipif(
    not CANONICAL_CORPUS.is_file() or not EVALUATION_DATASET.is_file(),
    reason="canonical evaluation data is managed by DVC",
)
def test_loads_real_corpus_grounded_dataset() -> None:
    dataset = load_evaluation_dataset(
        EVALUATION_DATASET,
        corpus_path=CANONICAL_CORPUS,
    )

    assert dataset.version == "legal-rag-eval-v1"
    assert len(dataset.cases) == 50
    assert all(isinstance(case, EvaluationCase) for case in dataset.cases)
    assert [case.id for case in dataset.cases] == [
        f"eval-ar-{number:03d}" for number in range(1, 51)
    ]


def test_valid_temporary_dataset_preserves_case_order(tmp_path: Path) -> None:
    cases = [
        case_record("case-b", 2, "نص المادة 2"),
        case_record("case-a", 1, "نص المادة 1"),
    ]
    cases[1]["question"] = "ما القاعدة القانونية الأخرى؟"
    dataset_path, corpus_path = temporary_files(tmp_path, cases)

    dataset = load_evaluation_dataset(dataset_path, corpus_path=corpus_path)

    assert [case.id for case in dataset.cases] == ["case-b", "case-a"]
    assert dataset.cases[0].relevant_article_numbers == (2,)


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    cases = [
        case_record("duplicate", 1, "نص المادة 1"),
        case_record("duplicate", 2, "نص المادة 2"),
    ]
    dataset_path, corpus_path = temporary_files(tmp_path, cases)

    with pytest.raises(ValueError, match="duplicate case IDs"):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)


def test_rejects_duplicate_questions(tmp_path: Path) -> None:
    cases = [
        case_record("case-1", 1, "نص المادة 1"),
        case_record("case-2", 2, "نص المادة 2"),
    ]
    cases[1]["question"] = cases[0]["question"]
    dataset_path, corpus_path = temporary_files(tmp_path, cases)

    with pytest.raises(ValueError, match="duplicate questions"):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)


def test_rejects_missing_article_reference(tmp_path: Path) -> None:
    case = case_record("missing", 99, "نص غير موجود")
    dataset_path, corpus_path = temporary_files(tmp_path, [case])
    write_json(corpus_path, [article_record(1, "نص المادة 1")])

    with pytest.raises(ValueError, match="references missing article 99"):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("question", 123, "question must be a non-empty string"),
        ("reference_answer", None, "reference_answer must be a non-empty string"),
        ("relevant_article_numbers", [True], "must contain integers"),
        ("language", "Arabic", "language must be 'ar' or 'en'"),
    ],
)
def test_rejects_invalid_schema_types(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    case = case_record("invalid", 1, "نص المادة 1")
    case[field] = value
    dataset_path, corpus_path = temporary_files(tmp_path, [case])

    with pytest.raises(ValueError, match=message):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)


def test_rejects_empty_relevant_articles(tmp_path: Path) -> None:
    case = case_record("empty-articles", 1, "نص المادة 1")
    case["relevant_article_numbers"] = []
    dataset_path, corpus_path = temporary_files(tmp_path, [case])

    with pytest.raises(ValueError, match="must be a non-empty list"):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)


def test_rejects_excerpt_not_found_in_canonical_article(tmp_path: Path) -> None:
    case = case_record("ungrounded", 1, "عبارة غير موجودة")
    dataset_path, corpus_path = temporary_files(tmp_path, [case])

    with pytest.raises(ValueError, match="excerpt is not present in article 1"):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)


def test_rejects_excerpt_article_order_mismatch(tmp_path: Path) -> None:
    case = case_record("mismatch", 1, "نص المادة 1")
    changed = deepcopy(case)
    changed["supporting_excerpts"][0]["article_number"] = 2
    dataset_path, corpus_path = temporary_files(tmp_path, [changed])

    with pytest.raises(ValueError, match="must exactly match"):
        load_evaluation_dataset(dataset_path, corpus_path=corpus_path)
