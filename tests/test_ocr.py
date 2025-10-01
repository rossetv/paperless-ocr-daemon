import os

import openai
import pytest
from PIL import Image

from paperless_ocr.config import Settings
from paperless_ocr.ocr import OpenAIProvider


@pytest.fixture
def settings(mocker):
    """Fixture to create a Settings object for tests."""
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "PRIMARY_MODEL": "gpt-primary",
            "FALLBACK_MODEL": "gpt-fallback",
            "MAX_RETRIES": "2",
        },
        clear=True,
    )
    return Settings()


@pytest.fixture
def ocr_provider(settings):
    """Fixture to create an OpenAIProvider instance."""
    return OpenAIProvider(settings)


@pytest.fixture
def mock_openai(mocker):
    """Fixture to mock the OpenAI API client."""
    return mocker.patch("openai.chat.completions.create")


def create_mock_response(mocker, content):
    """Helper to create a mock OpenAI API response."""
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = content
    mock_response = mocker.MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def create_test_image(blank=False):
    """Helper to create a test image."""
    if blank:
        return Image.new("RGB", (100, 100), "white")
    return Image.new("RGB", (100, 100), "black")


def test_transcribe_image_success(ocr_provider, mock_openai, mocker):
    """
    Test successful transcription with the primary model.
    """
    image = create_test_image()
    expected_text = "This is a test transcription."
    mock_openai.return_value = create_mock_response(mocker, expected_text)

    text, model = ocr_provider.transcribe_image(image)

    assert text == expected_text
    assert model == "gpt-primary"
    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "gpt-primary"


def test_transcribe_image_fallback_on_refusal(ocr_provider, mock_openai, mocker):
    """
    Test that the provider falls back to the secondary model if the primary refuses.
    """
    image = create_test_image()
    refusal_text = "I can't assist with that."
    expected_text = "Fallback model success."

    mock_openai.side_effect = [
        create_mock_response(mocker, refusal_text),
        create_mock_response(mocker, expected_text),
    ]

    text, model = ocr_provider.transcribe_image(image)

    assert text == expected_text
    assert model == "gpt-fallback"
    assert mock_openai.call_count == 2


def test_transcribe_image_all_models_refuse(
    ocr_provider, settings, mock_openai, mocker
):
    """
    Test that a refusal mark is returned if all models refuse.
    """
    image = create_test_image()
    refusal_text = "I can't assist with that."

    mock_openai.return_value = create_mock_response(mocker, refusal_text)

    text, model = ocr_provider.transcribe_image(image)

    assert text == settings.REFUSAL_MARK
    assert model == ""
    assert mock_openai.call_count == 2


def test_transcribe_blank_image(ocr_provider, mock_openai):
    """
    Test that blank images are skipped and the API is not called.
    """
    image = create_test_image(blank=True)

    text, model = ocr_provider.transcribe_image(image)

    assert text == ""
    assert model == ""
    mock_openai.assert_not_called()


def test_retry_on_api_error(ocr_provider, mock_openai, mocker):
    """
    Test that the provider retries on a transient API error.
    """
    image = create_test_image()
    expected_text = "Success after retry."

    mock_openai.side_effect = [
        openai.APIError("API is down", request=None, body=None),
        create_mock_response(mocker, expected_text),
    ]

    text, model = ocr_provider.transcribe_image(image)

    assert text == expected_text
    assert model == "gpt-primary"
    assert mock_openai.call_count == 2


def test_fallback_after_max_retries(ocr_provider, mock_openai, mocker):
    """
    Test that the provider falls back to the next model after exhausting all retries.
    """
    image = create_test_image()
    expected_text = "Fallback success."

    mock_openai.side_effect = [
        openai.APIError("API is down", request=None, body=None),
        openai.APIError("API is still down", request=None, body=None),
        create_mock_response(mocker, expected_text),  # Fallback model call
    ]

    text, model = ocr_provider.transcribe_image(image)

    assert text == expected_text
    assert model == "gpt-fallback"
    assert mock_openai.call_count == 3  # 2 for primary, 1 for fallback