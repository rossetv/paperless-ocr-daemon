import os

import httpx
import openai
import pytest

from classifier.provider import (
    ClassificationProvider,
    parse_classification_response,
)
from common.config import Settings


def create_mock_response(mocker, content):
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = content
    mock_response = mocker.MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def create_bad_request_error(message: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.com"))
    return openai.BadRequestError(message, response=response, body=None)


@pytest.fixture
def settings(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "AI_MODELS": "classify-primary,classify-fallback",
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
        "classifier.provider.ClassificationProvider._create_completion"
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


def test_classify_text_omits_temperature_for_gpt5(settings, mocker):
    settings.AI_MODELS = ["gpt-5-mini", "classify-fallback"]
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    mock_create.return_value = create_mock_response(
        mocker,
        '{"title":"Ok","correspondent":"","tags":[],"document_date":"2025-05-01",'
        '"document_type":"","language":"en","person":""}',
    )

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])

    assert result is not None
    assert model == "gpt-5-mini"
    _, kwargs = mock_create.call_args
    assert "temperature" not in kwargs


def test_classify_text_adds_max_tokens(settings, mocker):
    settings.AI_MODELS = ["classify-primary"]
    settings.CLASSIFY_MAX_TOKENS = 123
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    mock_create.return_value = create_mock_response(
        mocker,
        '{"title":"Ok","correspondent":"","tags":[],"document_date":"2025-05-01",'
        '"document_type":"","language":"en","person":""}',
    )

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])

    assert result is not None
    assert model == "classify-primary"
    _, kwargs = mock_create.call_args
    assert kwargs["max_tokens"] == 123


def test_classify_text_retries_without_temperature_on_error(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    temp_error = create_bad_request_error(
        "Unsupported value: 'temperature' does not support 0.2 with this model."
    )
    mock_create.side_effect = [
        temp_error,
        create_mock_response(
            mocker,
            '{"title":"Ok","correspondent":"","tags":[],"document_date":"2025-05-01",'
            '"document_type":"","language":"en","person":""}',
        ),
    ]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])

    assert result is not None
    assert model == "classify-primary"
    first_kwargs = mock_create.call_args_list[0].kwargs
    second_kwargs = mock_create.call_args_list[1].kwargs
    assert "temperature" in first_kwargs
    assert "temperature" not in second_kwargs
