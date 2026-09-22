"""Focused application configuration tests."""

import pytest

from legal_rag.config import AppConfig


def test_optional_llm_pricing_is_unconfigured_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_INPUT_COST_PER_1M_TOKENS_USD", raising=False)
    monkeypatch.delenv("LLM_OUTPUT_COST_PER_1M_TOKENS_USD", raising=False)

    config = AppConfig()

    assert config.llm_input_cost_per_million_tokens_usd is None
    assert config.llm_output_cost_per_million_tokens_usd is None


def test_llm_pricing_reads_explicit_nonnegative_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M_TOKENS_USD", "0.125")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M_TOKENS_USD", "0.5")

    config = AppConfig()

    assert config.llm_input_cost_per_million_tokens_usd == pytest.approx(0.125)
    assert config.llm_output_cost_per_million_tokens_usd == pytest.approx(0.5)


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_llm_pricing_rejects_invalid_rates(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M_TOKENS_USD", value)

    with pytest.raises(ValueError, match="finite nonnegative"):
        AppConfig()
