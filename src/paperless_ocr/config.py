"""
Configuration module for the Paperless-ngx OCR daemon.

This module centralizes the loading and validation of all configuration
parameters from environment variables. It provides a single `Settings`
class that acts as a container for all configurable values, ensuring
that they are defined in one place and can be easily imported and used
throughout the application.
"""

import os
from typing import Literal

import httpx
import openai
from PIL import Image


class Settings:
    """
    A container for all configuration settings, loaded from environment variables.

    This class centralizes configuration, providing default values for optional
    settings and raising errors for missing required settings.
    """

    # --- Paperless-ngx API Configuration ---
    PAPERLESS_URL: str
    PAPERLESS_TOKEN: str

    # --- LLM Provider Configuration ---
    LLM_PROVIDER: Literal["openai", "ollama"]
    OLLAMA_BASE_URL: str | None
    OPENAI_API_KEY: str | None

    # --- Model Selection ---
    PRIMARY_MODEL: str
    FALLBACK_MODEL: str

    # --- Paperless-ngx Tag IDs ---
    PRE_TAG_ID: int
    POST_TAG_ID: int

    # --- Classification Tag IDs ---
    CLASSIFY_PRE_TAG_ID: int
    CLASSIFY_POST_TAG_ID: int | None
    ERROR_TAG_ID: int | None

    # --- Daemon Configuration ---
    POLL_INTERVAL: int
    MAX_RETRIES: int
    MAX_RETRY_BACKOFF_SECONDS: int
    REQUEST_TIMEOUT: int

    # --- Image Processing Configuration ---
    OCR_DPI: int
    OCR_MAX_SIDE: int
    PAGE_WORKERS: int
    DOCUMENT_WORKERS: int

    # --- Logging ---
    LOG_LEVEL: str
    LOG_FORMAT: Literal["json", "console"]

    # --- Constants ---
    REFUSAL_MARK: str = "CHATGPT REFUSED TO TRANSCRIBE"

    # --- Classification Configuration ---
    CLASSIFY_MODEL: str
    CLASSIFY_FALLBACK_MODEL: str
    CLASSIFY_PERSON_FIELD_ID: int | None
    CLASSIFY_DEFAULT_COUNTRY_TAG: str
    CLASSIFY_MAX_CHARS: int
    CLASSIFY_TAG_LIMIT: int
    CLASSIFY_MAX_PAGES: int
    CLASSIFY_TAIL_PAGES: int
    CLASSIFY_HEADERLESS_CHAR_LIMIT: int

    def __init__(self):
        """
        Loads settings from environment variables and performs validation.
        """
        # --- Paperless-ngx API Configuration ---
        self.PAPERLESS_URL = os.getenv("PAPERLESS_URL", "http://paperless:8000").rstrip(
            "/"
        )
        self.PAPERLESS_TOKEN = self._get_required_env("PAPERLESS_TOKEN")

        # --- LLM Provider Configuration ---
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
        if self.LLM_PROVIDER not in ("openai", "ollama"):
            raise ValueError("LLM_PROVIDER must be 'openai' or 'ollama'")

        # --- Model Selection ---
        if self.LLM_PROVIDER == "ollama":
            self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
            self.OPENAI_API_KEY = None  # Not used for Ollama
            self.PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemma3:27b")
            self.FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemma3:12b")
        else:  # openai
            self.OLLAMA_BASE_URL = None
            self.OPENAI_API_KEY = self._get_required_env("OPENAI_API_KEY")
            self.PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-5-mini")
            self.FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "o4-mini")

        # --- Paperless-ngx Tag IDs ---
        self.PRE_TAG_ID = int(os.getenv("PRE_TAG_ID", "443"))
        self.POST_TAG_ID = int(os.getenv("POST_TAG_ID", "444"))

        # --- Classification Tag IDs ---
        self.CLASSIFY_PRE_TAG_ID = self._get_optional_int_env(
            "CLASSIFY_PRE_TAG_ID", self.POST_TAG_ID
        )
        self.CLASSIFY_POST_TAG_ID = self._get_optional_int_env("CLASSIFY_POST_TAG_ID")
        if self.CLASSIFY_POST_TAG_ID is not None and self.CLASSIFY_POST_TAG_ID <= 0:
            self.CLASSIFY_POST_TAG_ID = None
        self.ERROR_TAG_ID = self._get_optional_int_env("ERROR_TAG_ID", 552)
        if self.ERROR_TAG_ID is not None and self.ERROR_TAG_ID <= 0:
            self.ERROR_TAG_ID = None

        # --- Daemon Configuration ---
        self.POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "20"))
        if self.MAX_RETRIES < 1:
            raise ValueError("MAX_RETRIES must be >= 1")
        self.MAX_RETRY_BACKOFF_SECONDS = int(
            os.getenv("MAX_RETRY_BACKOFF_SECONDS", "30")
        )
        if self.MAX_RETRY_BACKOFF_SECONDS < 1:
            raise ValueError("MAX_RETRY_BACKOFF_SECONDS must be >= 1")
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

        # --- Image Processing Configuration ---
        self.OCR_DPI = int(os.getenv("OCR_DPI", "300"))
        self.OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1600"))
        self.PAGE_WORKERS = max(1, int(os.getenv("PAGE_WORKERS", "8")))
        self.DOCUMENT_WORKERS = max(1, int(os.getenv("DOCUMENT_WORKERS", "4")))

        # --- Logging ---
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        log_format = os.getenv("LOG_FORMAT", "console")
        if log_format not in ("json", "console"):
            raise ValueError("LOG_FORMAT must be 'json' or 'console'")
        self.LOG_FORMAT = log_format

        # --- Classification Configuration ---
        self.CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", self.PRIMARY_MODEL)
        self.CLASSIFY_FALLBACK_MODEL = os.getenv(
            "CLASSIFY_FALLBACK_MODEL", self.FALLBACK_MODEL
        )
        self.CLASSIFY_PERSON_FIELD_ID = self._get_optional_int_env(
            "CLASSIFY_PERSON_FIELD_ID"
        )
        self.CLASSIFY_DEFAULT_COUNTRY_TAG = os.getenv(
            "CLASSIFY_DEFAULT_COUNTRY_TAG", ""
        ).strip()
        self.CLASSIFY_MAX_CHARS = int(os.getenv("CLASSIFY_MAX_CHARS", "0"))
        self.CLASSIFY_TAG_LIMIT = max(0, int(os.getenv("CLASSIFY_TAG_LIMIT", "5")))
        self.CLASSIFY_MAX_PAGES = max(0, int(os.getenv("CLASSIFY_MAX_PAGES", "3")))
        self.CLASSIFY_TAIL_PAGES = max(0, int(os.getenv("CLASSIFY_TAIL_PAGES", "2")))
        self.CLASSIFY_HEADERLESS_CHAR_LIMIT = max(
            0, int(os.getenv("CLASSIFY_HEADERLESS_CHAR_LIMIT", "15000"))
        )

    def _get_required_env(self, var_name: str) -> str:
        """
        Gets a required environment variable, raising an error if it's not set.
        """
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Required environment variable '{var_name}' is not set.")
        return value

    def _get_optional_int_env(self, var_name: str, default: int | None = None) -> int | None:
        """
        Gets an optional int environment variable. Empty values return the default.
        """
        value = os.getenv(var_name)
        if value is None:
            return default
        value = value.strip()
        if not value:
            return default
        return int(value)


def setup_libraries(settings: Settings) -> None:
    """
    Configures third-party libraries based on the application settings.
    """
    # Disable Pillow's safety check that prevents huge images
    Image.MAX_IMAGE_PIXELS = None

    # Configure OpenAI SDK
    # The httpx library, used by OpenAI, automatically picks up proxy settings
    # from environment variables. In some environments, this can lead to unexpected
    # errors if the proxy is not configured correctly. To prevent this, we
    # temporarily unset the proxy environment variables before initializing the
    # http_client, ensuring that no proxy is used.
    original_proxies = (
        os.environ.pop("HTTP_PROXY", None),
        os.environ.pop("HTTPS_PROXY", None),
    )
    http_client = httpx.Client()
    # Restore original environment variables
    if original_proxies[0]:
        os.environ["HTTP_PROXY"] = original_proxies[0]
    if original_proxies[1]:
        os.environ["HTTPS_PROXY"] = original_proxies[1]

    if settings.LLM_PROVIDER == "ollama":
        openai.base_url = settings.OLLAMA_BASE_URL
        openai.api_key = "dummy"
    else:
        openai.api_key = settings.OPENAI_API_KEY

    openai.http_client = http_client
