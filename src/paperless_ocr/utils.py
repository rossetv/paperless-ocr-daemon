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
import logging
import random
import time
from functools import wraps
from typing import Callable, Type, TypeVar

from PIL import Image

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T")


def retry(
    retryable_exceptions: tuple[Type[Exception], ...],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator that retries a function call on specific exceptions.

    Args:
        retryable_exceptions: A tuple of exception types that should trigger a retry.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            settings: Settings = self.settings
            for attempt in range(1, settings.MAX_RETRIES + 1):
                try:
                    return func(self, *args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == settings.MAX_RETRIES:
                        log.exception(
                            "%s failed after %d attempts",
                            func.__name__,
                            attempt,
                        )
                        raise
                    log.warning(
                        "%s failed (%s) - retry %d/%d",
                        func.__name__,
                        e,
                        attempt,
                        settings.MAX_RETRIES,
                    )
                    _sleep_backoff(attempt, settings)
            # This part should be unreachable if MAX_RETRIES > 0
            raise RuntimeError("Retry loop exited unexpectedly.")

        return wrapper

    return decorator


def _sleep_backoff(attempt: int, settings: Settings) -> None:
    """Sleep for a short duration with exponential backoff and jitter."""
    delay = (2**attempt) * random.uniform(0.8, 1.2)
    log.info(
        "Sleeping %.1f s before retry %d/%d",
        delay,
        attempt,
        settings.MAX_RETRIES,
    )
    time.sleep(delay)


def is_blank(image: Image.Image, threshold: int = 5) -> bool:
    """
    Return True if the image is essentially blank (all white).
    """
    # Greyscale histogram - index 255 is pure white
    histogram = image.convert("L").histogram()
    return (sum(histogram) - histogram[255]) < threshold