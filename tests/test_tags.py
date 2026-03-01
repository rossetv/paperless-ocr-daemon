"""Tests for common.tags — tag extraction, hygiene, and lifecycle."""

from unittest.mock import MagicMock

from common.tags import (
    clean_pipeline_tags,
    extract_tags,
    get_latest_tags,
    release_processing_tag,
    remove_stale_queue_tag,
)


# ---------------------------------------------------------------------------
# extract_tags
# ---------------------------------------------------------------------------


def test_extract_tags_returns_set_from_list():
    doc = {"id": 1, "tags": [10, 20, 30]}
    assert extract_tags(doc, doc_id=1, context="test") == {10, 20, 30}


def test_extract_tags_handles_missing_key():
    assert extract_tags({}, doc_id=1, context="test") == set()


def test_extract_tags_handles_none_value():
    doc = {"tags": None}
    assert extract_tags(doc, doc_id=1, context="test") == set()


def test_extract_tags_handles_non_list_value():
    doc = {"tags": "not a list"}
    assert extract_tags(doc, doc_id=1, context="test") == set()


def test_extract_tags_handles_empty_list():
    doc = {"tags": []}
    assert extract_tags(doc, doc_id=1, context="test") == set()


# ---------------------------------------------------------------------------
# get_latest_tags
# ---------------------------------------------------------------------------


def test_get_latest_tags_returns_fresh_tags():
    client = MagicMock()
    client.get_document.return_value = {"id": 1, "tags": [10, 20]}
    result = get_latest_tags(client, doc_id=1)
    assert result == {10, 20}
    client.get_document.assert_called_once_with(1)


def test_get_latest_tags_falls_back_on_exception():
    client = MagicMock()
    client.get_document.side_effect = ConnectionError("network down")
    fallback = {"id": 1, "tags": [99]}
    result = get_latest_tags(client, doc_id=1, fallback_doc=fallback)
    assert result == {99}


def test_get_latest_tags_returns_empty_when_no_fallback():
    client = MagicMock()
    client.get_document.side_effect = ConnectionError("network down")
    result = get_latest_tags(client, doc_id=1)
    assert result == set()


# ---------------------------------------------------------------------------
# remove_stale_queue_tag
# ---------------------------------------------------------------------------


def test_remove_stale_queue_tag_removes_pre_tag():
    client = MagicMock()
    tags = {10, 20, 30}
    remove_stale_queue_tag(client, doc_id=1, tags=tags, pre_tag_id=10)
    args, kwargs = client.update_document_metadata.call_args
    assert 10 not in kwargs["tags"]
    assert 20 in kwargs["tags"]
    assert 30 in kwargs["tags"]


def test_remove_stale_queue_tag_removes_processing_tag():
    client = MagicMock()
    tags = {10, 20, 99}
    remove_stale_queue_tag(
        client, doc_id=1, tags=tags, pre_tag_id=10, processing_tag_id=99
    )
    args, kwargs = client.update_document_metadata.call_args
    assert 10 not in kwargs["tags"]
    assert 99 not in kwargs["tags"]
    assert 20 in kwargs["tags"]


def test_remove_stale_queue_tag_handles_api_error():
    client = MagicMock()
    client.update_document_metadata.side_effect = ConnectionError("fail")
    # Should not raise — errors are logged and swallowed
    remove_stale_queue_tag(client, doc_id=1, tags={10, 20}, pre_tag_id=10)


# ---------------------------------------------------------------------------
# release_processing_tag
# ---------------------------------------------------------------------------


def test_release_processing_tag_does_nothing_when_none():
    client = MagicMock()
    release_processing_tag(client, doc_id=1, tag_id=None, purpose="test")
    client.get_document.assert_not_called()


def test_release_processing_tag_does_nothing_when_zero():
    client = MagicMock()
    release_processing_tag(client, doc_id=1, tag_id=0, purpose="test")
    client.get_document.assert_not_called()


def test_release_processing_tag_removes_tag_when_present():
    client = MagicMock()
    client.get_document.return_value = {"id": 1, "tags": [10, 99]}
    release_processing_tag(client, doc_id=1, tag_id=99, purpose="test")
    args, kwargs = client.update_document_metadata.call_args
    assert 99 not in kwargs["tags"]
    assert 10 in kwargs["tags"]


def test_release_processing_tag_skips_update_when_tag_absent():
    client = MagicMock()
    client.get_document.return_value = {"id": 1, "tags": [10]}
    release_processing_tag(client, doc_id=1, tag_id=99, purpose="test")
    client.update_document_metadata.assert_not_called()


def test_release_processing_tag_handles_refresh_failure():
    client = MagicMock()
    client.get_document.side_effect = ConnectionError("fail")
    # Should not raise
    release_processing_tag(client, doc_id=1, tag_id=99, purpose="test")
    client.update_document_metadata.assert_not_called()


def test_release_processing_tag_handles_update_failure():
    client = MagicMock()
    client.get_document.return_value = {"id": 1, "tags": [10, 99]}
    client.update_document_metadata.side_effect = ConnectionError("fail")
    # Should not raise
    release_processing_tag(client, doc_id=1, tag_id=99, purpose="test")


# ---------------------------------------------------------------------------
# clean_pipeline_tags
# ---------------------------------------------------------------------------


def test_clean_pipeline_tags_removes_all_pipeline_tags():
    class FakeSettings:
        PRE_TAG_ID = 10
        POST_TAG_ID = 11
        CLASSIFY_PRE_TAG_ID = 12
        OCR_PROCESSING_TAG_ID = 13
        CLASSIFY_PROCESSING_TAG_ID = 14
        CLASSIFY_POST_TAG_ID = 15
        ERROR_TAG_ID = 16

    tags = {10, 11, 12, 13, 14, 15, 16, 42, 99}
    result = clean_pipeline_tags(tags, FakeSettings())
    assert result == {42, 99}


def test_clean_pipeline_tags_preserves_user_tags():
    class FakeSettings:
        PRE_TAG_ID = 10
        POST_TAG_ID = 11
        CLASSIFY_PRE_TAG_ID = 12
        OCR_PROCESSING_TAG_ID = None
        CLASSIFY_PROCESSING_TAG_ID = None
        CLASSIFY_POST_TAG_ID = None
        ERROR_TAG_ID = None

    tags = {10, 11, 12, 42, 99}
    result = clean_pipeline_tags(tags, FakeSettings())
    assert result == {42, 99}


def test_clean_pipeline_tags_handles_none_optional_tags():
    class FakeSettings:
        PRE_TAG_ID = 10
        POST_TAG_ID = 11
        CLASSIFY_PRE_TAG_ID = 12
        OCR_PROCESSING_TAG_ID = None
        CLASSIFY_PROCESSING_TAG_ID = None
        CLASSIFY_POST_TAG_ID = None
        ERROR_TAG_ID = None

    tags = {10, 42}
    result = clean_pipeline_tags(tags, FakeSettings())
    assert result == {42}


def test_clean_pipeline_tags_does_not_mutate_input():
    class FakeSettings:
        PRE_TAG_ID = 10
        POST_TAG_ID = 11
        CLASSIFY_PRE_TAG_ID = 12
        OCR_PROCESSING_TAG_ID = None
        CLASSIFY_PROCESSING_TAG_ID = None
        CLASSIFY_POST_TAG_ID = None
        ERROR_TAG_ID = None

    original = {10, 11, 42}
    result = clean_pipeline_tags(original, FakeSettings())
    assert result == {42}
    assert original == {10, 11, 42}  # unchanged
