import logging
import os

import structlog

from paperless_ocr.config import Settings
from paperless_ocr.logging_config import configure_logging


def _make_settings(log_format: str, log_level: str):
    os.environ.update(
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "LOG_FORMAT": log_format,
            "LOG_LEVEL": log_level,
        }
    )
    return Settings()


def _has_processor(formatter, processor_type):
    processors = getattr(formatter, "processors", None)
    if not processors:
        return False
    return any(isinstance(proc, processor_type) for proc in processors)


def test_configure_logging_console(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    settings = _make_settings(log_format="console", log_level="debug")

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    root_logger.handlers[:] = []
    try:
        configure_logging(settings)

        assert root_logger.getEffectiveLevel() == logging.DEBUG
        assert len(root_logger.handlers) == 1
        formatter = root_logger.handlers[0].formatter
        assert isinstance(formatter, structlog.stdlib.ProcessorFormatter)
        assert _has_processor(formatter, structlog.dev.ConsoleRenderer)
        assert logging.getLogger("httpx").level == logging.WARNING
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)


def test_configure_logging_json(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    settings = _make_settings(log_format="json", log_level="info")

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    root_logger.handlers[:] = []
    try:
        configure_logging(settings)

        assert root_logger.getEffectiveLevel() == logging.INFO
        assert len(root_logger.handlers) == 1
        formatter = root_logger.handlers[0].formatter
        assert isinstance(formatter, structlog.stdlib.ProcessorFormatter)
        assert _has_processor(formatter, structlog.processors.JSONRenderer)
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
