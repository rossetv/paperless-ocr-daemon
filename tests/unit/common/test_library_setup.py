"""
Comprehensive tests for ``common.library_setup.setup_libraries``.

Covers Pillow MAX_IMAGE_PIXELS, OpenAI provider configuration, Ollama
provider configuration, httpx client creation, and atexit registration.
"""

from __future__ import annotations

import atexit
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from PIL import Image

from common.library_setup import setup_libraries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    provider: str = "openai",
    api_key: str = "sk-test",
    base_url: str | None = None,
) -> MagicMock:
    s = MagicMock()
    s.LLM_PROVIDER = provider
    s.OPENAI_API_KEY = api_key
    s.OLLAMA_BASE_URL = base_url
    return s


@pytest.fixture(autouse=True)
def _reset_openai():
    """Save and restore openai module-level attributes."""
    orig_key = getattr(openai, "api_key", None)
    orig_base = getattr(openai, "base_url", None)
    orig_client = getattr(openai, "http_client", None)
    yield
    openai.api_key = orig_key
    openai.base_url = orig_base
    openai.http_client = orig_client


# ===================================================================
# Pillow MAX_IMAGE_PIXELS set to None
# ===================================================================

class TestPillowConfig:

    def test_max_image_pixels_set_to_none(self):
        settings = _make_settings()
        original = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 12345  # non-None sentinel
            setup_libraries(settings)
            assert Image.MAX_IMAGE_PIXELS is None
        finally:
            Image.MAX_IMAGE_PIXELS = original


# ===================================================================
# OpenAI provider configuration
# ===================================================================

class TestOpenAIProvider:

    def test_api_key_set(self):
        settings = _make_settings(provider="openai", api_key="sk-my-key")
        setup_libraries(settings)
        assert openai.api_key == "sk-my-key"

    def test_base_url_not_set(self):
        settings = _make_settings(provider="openai")
        old_base = openai.base_url
        setup_libraries(settings)
        # base_url should NOT be changed for openai provider
        # (the function does not set it for openai)
        assert openai.base_url == old_base

    def test_http_client_set(self):
        settings = _make_settings(provider="openai")
        setup_libraries(settings)
        assert openai.http_client is not None
        assert isinstance(openai.http_client, httpx.Client)


# ===================================================================
# Ollama provider configuration
# ===================================================================

class TestOllamaProvider:

    def test_base_url_set(self):
        settings = _make_settings(
            provider="ollama",
            base_url="http://ollama:11434/v1/",
        )
        setup_libraries(settings)
        assert openai.base_url == "http://ollama:11434/v1/"

    def test_api_key_set_to_dummy(self):
        settings = _make_settings(provider="ollama", base_url="http://localhost:11434/v1/")
        setup_libraries(settings)
        assert openai.api_key == "dummy"

    def test_http_client_set(self):
        settings = _make_settings(provider="ollama", base_url="http://localhost:11434/v1/")
        setup_libraries(settings)
        assert isinstance(openai.http_client, httpx.Client)


# ===================================================================
# httpx Client created with trust_env=False
# ===================================================================

class TestHttpxClient:

    @patch("common.library_setup.httpx.Client")
    @patch("common.library_setup.atexit.register")
    def test_trust_env_false(self, mock_register, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        settings = _make_settings()
        setup_libraries(settings)
        mock_client_cls.assert_called_once_with(trust_env=False)


# ===================================================================
# atexit.register called for cleanup
# ===================================================================

class TestAtexitRegistration:

    @patch("common.library_setup.atexit.register")
    @patch("common.library_setup.httpx.Client")
    def test_atexit_register_called(self, mock_client_cls, mock_register):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        settings = _make_settings()
        setup_libraries(settings)
        mock_register.assert_called_once_with(mock_instance.close)

    @patch("common.library_setup.atexit.register")
    @patch("common.library_setup.httpx.Client")
    def test_atexit_registers_client_close(self, mock_client_cls, mock_register):
        """The registered function should be the client's close method."""
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        settings = _make_settings(provider="ollama", base_url="http://localhost:11434/v1/")
        setup_libraries(settings)
        mock_register.assert_called_once_with(mock_instance.close)
