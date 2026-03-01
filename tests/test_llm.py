"""Tests for common.llm — shared LLM helpers."""

from common.llm import unique_models


def test_unique_models_deduplicates():
    result = unique_models(["a", "b", "a", "c", "b"])
    assert result == ["a", "b", "c"]


def test_unique_models_preserves_order():
    result = unique_models(["c", "b", "a"])
    assert result == ["c", "b", "a"]


def test_unique_models_empty():
    assert unique_models([]) == []


def test_unique_models_single():
    assert unique_models(["model-a"]) == ["model-a"]
