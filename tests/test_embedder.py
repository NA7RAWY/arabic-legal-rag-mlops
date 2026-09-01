"""Unit tests for sentence-transformer embedding behavior."""

import pytest

from legal_rag.ingestion import LegalChunk
from legal_rag.rag import SentenceTransformerEmbedder


class FakeArray:
    def __init__(self, values: list[float] | list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[float] | list[list[float]]:
        return self.values


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[str | list[str], bool]] = []

    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool,
    ) -> FakeArray:
        self.calls.append((sentences, normalize_embeddings))
        if isinstance(sentences, str):
            return FakeArray([0.25, 0.75])
        return FakeArray(
            [[float(index), float(index + 1)] for index, _ in enumerate(sentences)]
        )


@pytest.fixture
def fake_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SentenceTransformerEmbedder, FakeSentenceTransformer]:
    model = FakeSentenceTransformer()
    monkeypatch.setattr(
        SentenceTransformerEmbedder,
        "_load_model",
        lambda self: model,
    )
    return SentenceTransformerEmbedder("fake-model"), model


def _chunk(chunk_id: str, text: str) -> LegalChunk:
    article_number = int(chunk_id.removeprefix("article-"))
    return LegalChunk(
        chunk_id=chunk_id,
        article_number=article_number,
        text=text,
        language="ar",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=1,
        citation=f"Article {article_number}",
    )


def test_embed_documents_prefixes_normalizes_and_preserves_order(
    fake_embedder: tuple[SentenceTransformerEmbedder, FakeSentenceTransformer],
) -> None:
    embedder, model = fake_embedder

    vectors = embedder.embed_documents(["first", "second"])

    assert model.calls == [(["passage: first", "passage: second"], True)]
    assert vectors == [[0.0, 1.0], [1.0, 2.0]]
    assert all(isinstance(value, float) for vector in vectors for value in vector)


def test_embed_documents_empty_input_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = SentenceTransformerEmbedder("fake-model")
    monkeypatch.setattr(
        embedder,
        "_load_model",
        lambda: pytest.fail("model should not be loaded"),
    )

    assert embedder.embed_documents([]) == []


def test_embed_query_prefixes_normalizes_and_returns_list(
    fake_embedder: tuple[SentenceTransformerEmbedder, FakeSentenceTransformer],
) -> None:
    embedder, model = fake_embedder

    vector = embedder.embed_query("ما هو العقد؟")

    assert model.calls == [("query: ما هو العقد؟", True)]
    assert vector == [0.25, 0.75]
    assert isinstance(vector, list)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_embed_query_rejects_empty_text(query: str) -> None:
    embedder = SentenceTransformerEmbedder("fake-model")

    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed_query(query)


def test_embed_chunks_preserves_chunk_order(
    fake_embedder: tuple[SentenceTransformerEmbedder, FakeSentenceTransformer],
) -> None:
    embedder, model = fake_embedder
    chunks = [_chunk("article-9", "nine"), _chunk("article-2", "two")]

    vectors = embedder.embed_chunks(chunks)

    assert model.calls == [(["passage: nine", "passage: two"], True)]
    assert vectors == [[0.0, 1.0], [1.0, 2.0]]
