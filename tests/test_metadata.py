"""Tests for classifier.metadata — date, language, and custom-field helpers."""

import datetime as dt

from classifier.metadata import (
    is_empty_classification,
    normalize_language,
    parse_document_date,
    resolve_date_for_tags,
    update_custom_fields,
)
from classifier.result import ClassificationResult


# ---------------------------------------------------------------------------
# parse_document_date
# ---------------------------------------------------------------------------


def test_parse_document_date_valid_iso():
    assert parse_document_date("2024-03-15") == "2024-03-15"


def test_parse_document_date_with_time_component():
    assert parse_document_date("2024-03-15T14:30:00") == "2024-03-15"


def test_parse_document_date_empty():
    assert parse_document_date("") is None


def test_parse_document_date_none_like():
    assert parse_document_date(None) is None


def test_parse_document_date_invalid():
    assert parse_document_date("not-a-date") is None


def test_parse_document_date_strips_whitespace():
    assert parse_document_date("  2024-03-15  ") == "2024-03-15"


# ---------------------------------------------------------------------------
# resolve_date_for_tags
# ---------------------------------------------------------------------------


def test_resolve_date_prefers_result_date():
    assert resolve_date_for_tags("2024-06-01", "2023-01-01") == "2024-06-01"


def test_resolve_date_falls_back_to_existing():
    assert resolve_date_for_tags(None, "2023-01-01") == "2023-01-01"


def test_resolve_date_falls_back_to_existing_with_time():
    assert resolve_date_for_tags(None, "2023-01-01T12:00:00Z") == "2023-01-01"


def test_resolve_date_returns_today_when_all_empty():
    result = resolve_date_for_tags(None, None)
    assert result == dt.date.today().isoformat()


def test_resolve_date_skips_invalid_values():
    result = resolve_date_for_tags("invalid", "also-invalid")
    assert result == dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# normalize_language
# ---------------------------------------------------------------------------


def test_normalize_language_two_letter():
    assert normalize_language("en") == "en"
    assert normalize_language("DE") == "de"


def test_normalize_language_locale_style():
    assert normalize_language("en-US") == "en"
    assert normalize_language("pt_BR") == "pt"


def test_normalize_language_undetermined():
    assert normalize_language("und") == "und"


def test_normalize_language_empty():
    assert normalize_language("") is None


def test_normalize_language_invalid_falls_back():
    assert normalize_language("english") == "und"


def test_normalize_language_strips_whitespace():
    assert normalize_language("  fr  ") == "fr"


# ---------------------------------------------------------------------------
# update_custom_fields
# ---------------------------------------------------------------------------


def test_update_custom_fields_upsert_existing():
    existing = [{"field": 1, "value": "old"}, {"field": 2, "value": "keep"}]
    result = update_custom_fields(existing, field_id=1, value="new")
    assert result == [{"field": 1, "value": "new"}, {"field": 2, "value": "keep"}]


def test_update_custom_fields_append_new():
    existing = [{"field": 1, "value": "a"}]
    result = update_custom_fields(existing, field_id=2, value="b")
    assert result == [{"field": 1, "value": "a"}, {"field": 2, "value": "b"}]


def test_update_custom_fields_none_existing():
    result = update_custom_fields(None, field_id=1, value="new")
    assert result == [{"field": 1, "value": "new"}]


def test_update_custom_fields_does_not_mutate_original():
    existing = [{"field": 1, "value": "old"}]
    result = update_custom_fields(existing, field_id=1, value="new")
    assert existing[0]["value"] == "old"
    assert result[0]["value"] == "new"


# ---------------------------------------------------------------------------
# is_empty_classification
# ---------------------------------------------------------------------------


def test_is_empty_classification_all_empty():
    result = ClassificationResult(
        title="", correspondent="", tags=[], document_date="",
        document_type="", language="", person="",
    )
    assert is_empty_classification(result) is True


def test_is_empty_classification_has_title():
    result = ClassificationResult(
        title="Invoice", correspondent="", tags=[], document_date="",
        document_type="", language="", person="",
    )
    assert is_empty_classification(result) is False


def test_is_empty_classification_has_tags():
    result = ClassificationResult(
        title="", correspondent="", tags=["Bills"], document_date="",
        document_type="", language="", person="",
    )
    assert is_empty_classification(result) is False


def test_is_empty_classification_whitespace_only():
    result = ClassificationResult(
        title="  ", correspondent="  ", tags=["  "], document_date="",
        document_type="", language="", person="",
    )
    assert is_empty_classification(result) is True


def test_is_empty_classification_none_fields_safe():
    """Ensure None fields don't crash (defensive dataclass construction)."""
    result = ClassificationResult(
        title=None, correspondent=None, tags=None, document_date=None,
        document_type=None, language=None, person=None,
    )
    assert is_empty_classification(result) is True


def test_is_empty_classification_has_person():
    result = ClassificationResult(
        title="", correspondent="", tags=[], document_date="",
        document_type="", language="", person="John",
    )
    assert is_empty_classification(result) is False
