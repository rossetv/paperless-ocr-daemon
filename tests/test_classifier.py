import os

import pytest

from paperless_ocr.classifier import ClassificationProvider, parse_classification_response
from paperless_ocr.config import Settings


def create_mock_response(mocker, content):
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = content
    mock_response = mocker.MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def settings(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "CLASSIFY_MODEL": "classify-primary",
            "CLASSIFY_FALLBACK_MODEL": "classify-fallback",
        },
        clear=True,
    )
    settings_obj = Settings()
    settings_obj.MAX_RETRIES = 1
    return settings_obj


def test_parse_classification_response_basic():
    raw = (
        '{"title":"Test","correspondent":"ACME","tags":["A","B"],'
        '"document_date":"2025-05-01","document_type":"Invoice","language":"en",'
        '"person":"John Doe"}'
    )

    result = parse_classification_response(raw)

    assert result.title == "Test"
    assert result.correspondent == "ACME"
    assert result.tags == ["A", "B"]
    assert result.document_date == "2025-05-01"
    assert result.document_type == "Invoice"
    assert result.language == "en"
    assert result.person == "John Doe"


def test_parse_classification_response_tags_string():
    raw = (
        '{"title":"","correspondent":"","tags":"Bills","document_date":"",'
        '"document_type":"","language":"","person":""}'
    )

    result = parse_classification_response(raw)

    assert result.tags == ["Bills"]


def test_classify_text_fallback_on_invalid_json(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "paperless_ocr.classifier.ClassificationProvider._create_completion"
    )
    mock_create.side_effect = [
        create_mock_response(mocker, "not json"),
        create_mock_response(
            mocker,
            '{"title":"Ok","correspondent":"","tags":[],"document_date":"2025-05-01",'
            '"document_type":"","language":"en","person":""}',
        ),
    ]

    result, model = provider.classify_text(
        "text",
        correspondents=["A"],
        document_types=["B"],
        tags=["C"],
    )

    assert result is not None
    assert model == "classify-fallback"
