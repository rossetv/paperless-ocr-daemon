import os

import pytest
from paperless_ocr.config import Settings, setup_libraries


def test_settings_default_values(mocker):
    """
    Test that the Settings class loads default values correctly when no
    environment variables are set.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
        },
        clear=True,
    )

    settings = Settings()

    assert settings.PAPERLESS_URL == "http://paperless:8000"
    assert settings.LLM_PROVIDER == "openai"
    assert settings.PRIMARY_MODEL == "gpt-5-mini"
    assert settings.POLL_INTERVAL == 15
    assert settings.OCR_DPI == 300


def test_settings_from_environment_variables(mocker):
    """
    Test that the Settings class correctly loads values from environment variables.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_URL": "http://test-paperless:1234/",
            "PAPERLESS_TOKEN": "env_token",
            "OPENAI_API_KEY": "env_api_key",
            "LLM_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://ollama:11434/v1",
            "PRIMARY_MODEL": "test-primary-model",
            "PRE_TAG_ID": "101",
            "POLL_INTERVAL": "30",
        },
        clear=True,
    )

    settings = Settings()

    assert settings.PAPERLESS_URL == "http://test-paperless:1234"
    assert settings.PAPERLESS_TOKEN == "env_token"
    assert settings.LLM_PROVIDER == "ollama"
    assert settings.OLLAMA_BASE_URL == "http://ollama:11434/v1"
    assert settings.PRIMARY_MODEL == "test-primary-model"
    assert settings.PRE_TAG_ID == 101
    assert settings.POLL_INTERVAL == 30


def test_settings_missing_required_env_vars(mocker):
    """
    Test that the Settings class raises a ValueError if a required environment
    variable is not set.
    """
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(
        ValueError, match="Required environment variable 'PAPERLESS_TOKEN' is not set."
    ):
        Settings()

    mocker.patch.dict(os.environ, {"PAPERLESS_TOKEN": "some-token"}, clear=True)

    with pytest.raises(
        ValueError, match="Required environment variable 'OPENAI_API_KEY' is not set."
    ):
        Settings()


def test_ollama_configuration(mocker):
    """
    Test that the settings for the Ollama provider are configured correctly.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "LLM_PROVIDER": "ollama",
        },
        clear=True,
    )

    settings = Settings()
    setup_libraries(settings)  # This should configure the openai client

    import openai

    assert settings.LLM_PROVIDER == "ollama"
    assert settings.PRIMARY_MODEL == "gemma3:27b"
    assert openai.base_url == "http://localhost:11434/v1/"
    assert openai.api_key == "dummy"


def test_invalid_llm_provider(mocker):
    """
    Test that an invalid LLM_PROVIDER value raises a ValueError.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "LLM_PROVIDER": "invalid_provider",
        },
        clear=True,
    )

    with pytest.raises(ValueError, match="LLM_PROVIDER must be 'openai' or 'ollama'"):
        Settings()


def test_invalid_log_format(mocker):
    """
    Test that an invalid LOG_FORMAT value raises a ValueError.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "LOG_FORMAT": "xml",
        },
        clear=True,
    )

    with pytest.raises(ValueError, match="LOG_FORMAT must be 'json' or 'console'"):
        Settings()


def test_minimum_worker_counts(mocker):
    """
    Test that worker counts are clamped to a minimum of 1.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "PAGE_WORKERS": "0",
            "DOCUMENT_WORKERS": "-5",
        },
        clear=True,
    )

    settings = Settings()

    assert settings.PAGE_WORKERS == 1
    assert settings.DOCUMENT_WORKERS == 1


def test_setup_libraries_restores_proxies(mocker):
    """
    Test that proxy environment variables are restored after setup.
    """
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "HTTP_PROXY": "http://proxy.local",
            "HTTPS_PROXY": "https://proxy.local",
        },
        clear=True,
    )

    settings = Settings()
    setup_libraries(settings)

    assert os.environ.get("HTTP_PROXY") == "http://proxy.local"
    assert os.environ.get("HTTPS_PROXY") == "https://proxy.local"
