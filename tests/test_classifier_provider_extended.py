"""Extended tests for classifier.provider — compatibility handling."""

import json
import os

import httpx
import openai
import pytest

from classifier.provider import ClassificationProvider
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


def create_api_error(message: str) -> openai.APIError:
    request = httpx.Request("POST", "https://example.com")
    return openai.APIError(message, request=request, body=None)


@pytest.fixture
def settings(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "AI_MODELS": "primary-model,fallback-model",
        },
        clear=True,
    )
    s = Settings()
    s.MAX_RETRIES = 1
    return s


VALID_JSON = json.dumps({
    "title": "Ok", "correspondent": "", "tags": [],
    "document_date": "2025-05-01", "document_type": "Invoice",
    "language": "en", "person": "",
})


def test_response_format_error_retries_without_it(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    err = create_bad_request_error("response_format is not supported for this model")
    mock_create.side_effect = [err, create_mock_response(mocker, VALID_JSON)]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])
    assert result is not None
    assert model == "primary-model"


def test_max_tokens_error_retries_without_it(settings, mocker):
    settings.CLASSIFY_MAX_TOKENS = 100
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    err = create_bad_request_error("max_tokens is not supported")
    mock_create.side_effect = [err, create_mock_response(mocker, VALID_JSON)]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])
    assert result is not None


def test_api_error_falls_through_to_next_model(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    err = create_api_error("server error")
    mock_create.side_effect = [err, create_mock_response(mocker, VALID_JSON)]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])
    assert result is not None
    assert model == "fallback-model"


def test_all_models_fail_returns_none(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    err = create_api_error("server error")
    mock_create.side_effect = [err, err]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])
    assert result is None
    assert model == ""


def test_empty_text_returns_none(settings):
    provider = ClassificationProvider(settings)
    result, model = provider.classify_text("", ["A"], ["B"], ["C"])
    assert result is None
    assert model == ""


def test_whitespace_text_returns_none(settings):
    provider = ClassificationProvider(settings)
    result, model = provider.classify_text("   ", ["A"], ["B"], ["C"])
    assert result is None
    assert model == ""


def test_bad_request_unrelated_falls_through(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    err = create_bad_request_error("some other bad request error")
    mock_create.side_effect = [err, create_mock_response(mocker, VALID_JSON)]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])
    # First model returned None, fallback succeeds
    assert result is not None
    assert model == "fallback-model"


def test_response_format_for_ollama(settings, mocker):
    settings.LLM_PROVIDER = "ollama"
    provider = ClassificationProvider(settings)
    assert provider._response_format() is None


def test_response_format_for_openai(settings, mocker):
    settings.LLM_PROVIDER = "openai"
    provider = ClassificationProvider(settings)
    fmt = provider._response_format()
    assert fmt is not None
    assert fmt["type"] == "json_schema"


def test_truncation_note_passed_to_user_message(settings, mocker):
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    mock_create.return_value = create_mock_response(mocker, VALID_JSON)

    provider.classify_text(
        "text", ["A"], ["B"], ["C"],
        truncation_note="Document was truncated.",
    )

    call_kwargs = mock_create.call_args[1]
    user_msg = call_kwargs["messages"][1]["content"]
    assert "Document was truncated." in user_msg


def test_get_stats_returns_copy(settings):
    provider = ClassificationProvider(settings)
    provider._reset_stats()
    stats = provider.get_stats()
    stats["attempts"] = 999
    assert provider.get_stats()["attempts"] == 0


def test_content_none_handled(settings, mocker):
    """When API returns content=None, it should not crash."""
    provider = ClassificationProvider(settings)
    mock_create = mocker.patch(
        "classifier.provider.ClassificationProvider._create_completion"
    )
    mock_resp = create_mock_response(mocker, None)
    mock_resp.choices[0].message.content = None
    good_resp = create_mock_response(mocker, VALID_JSON)
    mock_create.side_effect = [mock_resp, good_resp]

    result, model = provider.classify_text("text", ["A"], ["B"], ["C"])
    # None content on primary → parse error → fallback succeeds
    assert result is not None
    assert model == "fallback-model"
