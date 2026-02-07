"""
Utilities
=========

This module provides utility functions and classes that are used across
the application but do not belong to a more specific domain like the
Paperless-ngx client or the OCR logic.

Currently, it contains a robust `retry` decorator for handling transient
errors with exponential backoff and jitter, as well as a helper for
detecting blank images. These utilities help to improve the resilience
and efficiency of the daemon.
"""
import random
import re
import time
from functools import wraps
from typing import Callable, Iterable, Type, TypeVar

import structlog
from PIL import Image

from .config import Settings

log = structlog.get_logger(__name__)
T = TypeVar("T")
_REDACTED_RE = re.compile(r"\[[^\]]*redacted[^\]]*\]", re.IGNORECASE)


def retry(
    retryable_exceptions: tuple[Type[Exception], ...],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator that retries a function call on specific exceptions.

    Args:
        retryable_exceptions: A tuple of exception types that should trigger a retry.

    Notes:
        The wrapped method must be on an object with a ``settings`` attribute
        that provides ``MAX_RETRIES`` and ``MAX_RETRY_BACKOFF_SECONDS``.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            settings: Settings = self.settings
            if settings.MAX_RETRIES < 1:
                raise ValueError("MAX_RETRIES must be >= 1")
            for attempt in range(1, settings.MAX_RETRIES + 1):
                try:
                    return func(self, *args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == settings.MAX_RETRIES:
                        log.exception(
                            "Function failed after all retries",
                            func_name=func.__name__,
                            attempt=attempt,
                        )
                        raise
                    log.warning(
                        "Function failed, retrying",
                        func_name=func.__name__,
                        error=e,
                        attempt=attempt,
                        max_retries=settings.MAX_RETRIES,
                    )
                    _sleep_backoff(attempt, settings)
            # This part should be unreachable if MAX_RETRIES > 0
            raise RuntimeError("Retry loop exited unexpectedly.")

        return wrapper

    return decorator


def _sleep_backoff(attempt: int, settings: Settings) -> None:
    """Sleep for a short duration with exponential backoff, jitter, and a cap."""
    delay = (2**attempt) * random.uniform(0.8, 1.2)
    delay = min(delay, settings.MAX_RETRY_BACKOFF_SECONDS)
    log.info(
        "Sleeping before retry",
        delay=f"{delay:.1f}s",
        attempt=attempt,
        max_retries=settings.MAX_RETRIES,
        max_backoff=settings.MAX_RETRY_BACKOFF_SECONDS,
    )
    time.sleep(delay)


def is_blank(image: Image.Image, threshold: int = 5) -> bool:
    """
    Return True if the image is essentially blank (all white).
    """
    # Greyscale histogram - index 255 is pure white
    histogram = image.convert("L").histogram()
    return (sum(histogram) - histogram[255]) < threshold


def contains_redacted_marker(text: str) -> bool:
    """
    Return True if the text includes bracketed redaction markers.

    Examples: "[REDACTED]", "[REDACTED NAME]", "[NAME REDACTED]".
    """
    return bool(_REDACTED_RE.search(text))


def is_error_content(text: str, error_phrases: Iterable[str]) -> bool:
    """
    Return True if the text contains refusal phrases or redaction markers.
    """
    text_lower = text.lower()
    return contains_redacted_marker(text) or any(
        phrase in text_lower for phrase in error_phrases
    )
