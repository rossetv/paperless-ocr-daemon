import os
from unittest.mock import MagicMock

import pytest

from paperless_ocr import classify_worker
from paperless_ocr.classify_worker import ClassificationProcessor, enrich_tags, truncate_content_by_pages
from paperless_ocr.config import Settings


def test_enrich_tags_adds_required_tags():
    text = (
        "I'm sorry, I can't assist with that.\n"
        "Transcribed by model: gpt-5-mini"
    )
    tags = ["Bills"]
    result = enrich_tags(tags, text, "2024-03-01", "Ireland", 8)

    assert "Bills" in result
    assert "ERROR" in result
    assert "gpt-5-mini" in result
    assert "2024" in result
    assert "Ireland" in result


def test_enrich_tags_trims_to_limit():
    text = "Transcribed by model: gpt-5"
    tags = ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5", "Tag6", "Tag7", "Tag8"]
    result = enrich_tags(tags, text, "2024-01-01", "Ireland", 4)

    assert len(result) == 7
    assert "gpt-5" in result
    assert "2024" in result
    assert "Ireland" in result


def test_enrich_tags_extracts_multiple_models():
    text = "Content\n\nTranscribed by model: gpt-5-mini, o4-mini"
    result = enrich_tags([], text, "2024-01-01", "", 8)

    assert "gpt-5-mini" in result
    assert "o4-mini" in result


def test_truncate_content_by_pages_limits_pages_and_keeps_footer():
    content = (
        "--- Page 1 ---\n"
        "Page1\n\n"
        "--- Page 2 ---\n"
        "Page2\n\n"
        "--- Page 3 ---\n"
        "Page3\n\n"
        "\n\nTranscribed by model: gpt-5-mini"
    )

    result, note = truncate_content_by_pages(content, 2, 0, 1000)

    assert "--- Page 1 ---" in result
    assert "--- Page 2 ---" in result
    assert "--- Page 3 ---" not in result
    assert "Transcribed by model: gpt-5-mini" in result
    assert note is not None


def test_truncate_content_by_pages_no_headers_returns_full():
    content = "Single page content"

    result, note = truncate_content_by_pages(content, 5, 0, 1000)

    assert result == content
    assert note is None


def test_truncate_content_by_pages_no_headers_truncates_by_chars():
    content = "A" * (200 + 10)

    result, note = truncate_content_by_pages(content, 5, 0, 200)

    assert len(result) == 200
    assert note is not None


def test_truncate_content_by_pages_includes_tail_pages():
    content = (
        "--- Page 1 ---\nA\n\n"
        "--- Page 2 ---\nB\n\n"
        "--- Page 3 ---\nC\n\n"
        "--- Page 4 ---\nD\n\n"
        "--- Page 5 ---\nE\n\n"
        "--- Page 6 ---\nF\n\n"
        "\n\nTranscribed by model: gpt-5-mini"
    )

    result, note = truncate_content_by_pages(content, 3, 2, 1000)

    assert "--- Page 1 ---" in result
    assert "--- Page 2 ---" in result
    assert "--- Page 3 ---" in result
    assert "--- Page 4 ---" not in result
    assert "--- Page 5 ---" in result
    assert "--- Page 6 ---" in result
    assert note is not None


@pytest.fixture
def settings(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "PRE_TAG_ID": "443",
            "POST_TAG_ID": "444",
            "ERROR_TAG_ID": "552",
        },
        clear=True,
    )
    return Settings()


def test_skips_classification_when_error_tag_present(settings):
    doc = {"id": 1, "title": "Doc", "tags": [settings.ERROR_TAG_ID, 99]}
    paperless_client = MagicMock()
    paperless_client.get_document.return_value = {
        "id": 1,
        "content": "text",
        "tags": [settings.ERROR_TAG_ID, settings.PRE_TAG_ID, settings.POST_TAG_ID, 99],
    }
    classifier = MagicMock()
    taxonomy_cache = MagicMock()

    processor = ClassificationProcessor(
        doc,
        paperless_client,
        classifier,
        taxonomy_cache,
        settings,
    )

    processor.process()

    paperless_client.update_document_metadata.assert_called_once()
    args, kwargs = paperless_client.update_document_metadata.call_args
    assert args[0] == 1
    assert set(kwargs["tags"]) == {settings.ERROR_TAG_ID, 99}
    classifier.classify_text.assert_not_called()


def test_requeues_document_when_content_empty(settings):
    doc = {"id": 2, "title": "Doc", "tags": [settings.POST_TAG_ID, 77]}
    paperless_client = MagicMock()
    paperless_client.get_document.return_value = {
        "id": 2,
        "content": "",
        "tags": [settings.POST_TAG_ID, 77],
    }
    classifier = MagicMock()
    taxonomy_cache = MagicMock()

    processor = ClassificationProcessor(
        doc,
        paperless_client,
        classifier,
        taxonomy_cache,
        settings,
    )

    processor.process()

    paperless_client.update_document_metadata.assert_called_once()
    args, kwargs = paperless_client.update_document_metadata.call_args
    assert args[0] == 2
    assert set(kwargs["tags"]) == {settings.PRE_TAG_ID, 77}
    classifier.classify_text.assert_not_called()
