"""Tests for classifier.result — JSON parsing and ClassificationResult."""

import json

import pytest

from classifier.result import (
    ClassificationResult,
    _extract_json,
    parse_classification_response,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_clean():
    data = _extract_json('{"key": "value"}')
    assert data == {"key": "value"}


def test_extract_json_markdown_fences():
    raw = '```json\n{"key": "value"}\n```'
    data = _extract_json(raw)
    assert data == {"key": "value"}


def test_extract_json_preamble_text():
    raw = 'Here is the result:\n{"key": "value"}\nDone.'
    data = _extract_json(raw)
    assert data == {"key": "value"}


def test_extract_json_no_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("no json here")


def test_extract_json_empty_braces():
    data = _extract_json("{}")
    assert data == {}


# ---------------------------------------------------------------------------
# parse_classification_response
# ---------------------------------------------------------------------------


def test_parse_classification_response_full():
    raw = json.dumps({
        "title": "Invoice 123",
        "correspondent": "ACME Corp",
        "tags": ["Bills", "2024"],
        "document_date": "2024-01-15",
        "document_type": "Invoice",
        "language": "en",
        "person": "Jane Doe",
    })
    result = parse_classification_response(raw)
    assert result.title == "Invoice 123"
    assert result.correspondent == "ACME Corp"
    assert result.tags == ["Bills", "2024"]
    assert result.document_date == "2024-01-15"
    assert result.document_type == "Invoice"
    assert result.language == "en"
    assert result.person == "Jane Doe"


def test_parse_classification_response_null_values():
    raw = json.dumps({
        "title": None,
        "correspondent": None,
        "tags": None,
        "document_date": None,
        "document_type": None,
        "language": None,
        "person": None,
    })
    result = parse_classification_response(raw)
    assert result.title == ""
    assert result.correspondent == ""
    assert result.tags == []
    assert result.document_date == ""
    assert result.document_type == ""
    assert result.language == ""
    assert result.person == ""


def test_parse_classification_response_tags_as_string():
    raw = json.dumps({
        "title": "",
        "correspondent": "",
        "tags": "Bills",
        "document_date": "",
        "document_type": "",
        "language": "",
        "person": "",
    })
    result = parse_classification_response(raw)
    assert result.tags == ["Bills"]


def test_parse_classification_response_tags_as_empty_string():
    raw = json.dumps({
        "title": "",
        "correspondent": "",
        "tags": "  ",
        "document_date": "",
        "document_type": "",
        "language": "",
        "person": "",
    })
    result = parse_classification_response(raw)
    assert result.tags == []


def test_parse_classification_response_tags_invalid_type():
    raw = json.dumps({
        "title": "",
        "correspondent": "",
        "tags": 42,
        "document_date": "",
        "document_type": "",
        "language": "",
        "person": "",
    })
    result = parse_classification_response(raw)
    assert result.tags == []


def test_parse_classification_response_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_classification_response("")


def test_parse_classification_response_not_dict_raises():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_classification_response("[1, 2, 3]")


def test_parse_classification_response_frozen():
    raw = json.dumps({
        "title": "T", "correspondent": "", "tags": [],
        "document_date": "", "document_type": "", "language": "", "person": "",
    })
    result = parse_classification_response(raw)
    with pytest.raises(AttributeError):
        result.title = "changed"


def test_parse_classification_response_strips_tag_whitespace():
    raw = json.dumps({
        "title": "", "correspondent": "", "tags": ["  Bills  ", "  "],
        "document_date": "", "document_type": "", "language": "", "person": "",
    })
    result = parse_classification_response(raw)
    assert result.tags == ["Bills"]
