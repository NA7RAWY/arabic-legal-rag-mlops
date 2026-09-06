"""Print a deterministic summary of the legal RAG evaluation dataset."""

from __future__ import annotations

from collections import Counter

from legal_rag.config import get_config
from legal_rag.evaluation.dataset import (
    DEFAULT_EVALUATION_PATH,
    load_evaluation_dataset,
)


def main() -> None:
    """Validate the default dataset and print summary counts."""

    config = get_config()
    dataset = load_evaluation_dataset(
        DEFAULT_EVALUATION_PATH,
        corpus_path=config.corpus_path,
    )
    languages = Counter(case.language for case in dataset.cases)
    categories = Counter(case.category for case in dataset.cases)
    single_article = sum(
        len(case.relevant_article_numbers) == 1 for case in dataset.cases
    )
    referenced_articles = sorted(
        {
            article_number
            for case in dataset.cases
            for article_number in case.relevant_article_numbers
        }
    )
    article_frequencies = Counter(
        article_number
        for case in dataset.cases
        for article_number in case.relevant_article_numbers
    )

    print(f"Dataset version: {dataset.version}")
    print(f"Cases: {len(dataset.cases)}")
    print("Languages:")
    for language, count in sorted(languages.items()):
        print(f"  {language}: {count}")
    print("Categories:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    print(f"Single-article cases: {single_article}")
    print(f"Multi-article cases: {len(dataset.cases) - single_article}")
    print(f"Unique referenced articles: {len(referenced_articles)}")
    print("Referenced article numbers: " + ", ".join(map(str, referenced_articles)))
    print("Article frequencies:")
    for article_number, count in sorted(
        article_frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {article_number}: {count}")


if __name__ == "__main__":
    main()
