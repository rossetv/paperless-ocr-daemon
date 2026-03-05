"""
LLM Concurrency Limiter
========================

Provides a semaphore that caps the number of concurrent LLM API calls
across all worker threads.  This is especially useful for self-hosted Ollama
deployments where the GPU is the bottleneck.

Typical usage::

    from common.concurrency import llm_limiter

    llm_limiter.init(4)               # once at startup

    with llm_limiter.acquire():        # in any worker thread
        openai.chat.completions.create(...)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator

import structlog

log = structlog.get_logger(__name__)


class LLMConcurrencyLimiter:
    """Encapsulates a concurrency semaphore for LLM API calls."""

    def __init__(self) -> None:
        self._semaphore: threading.Semaphore | None = None

    def init(self, max_concurrent: int) -> None:
        """Initialize the LLM concurrency semaphore.

        Args:
            max_concurrent: Maximum concurrent LLM calls.  ``0`` means unlimited.
        """
        if max_concurrent > 0:
            self._semaphore = threading.BoundedSemaphore(max_concurrent)
            log.info("LLM concurrency limiter enabled", max_concurrent=max_concurrent)
        else:
            self._semaphore = None

    @contextmanager
    def acquire(self) -> Generator[None, None, None]:
        """Context manager that acquires the LLM concurrency semaphore.

        When ``LLM_MAX_CONCURRENT`` is ``0`` (unlimited), this is a no-op.
        """
        if self._semaphore is not None:
            self._semaphore.acquire()
            try:
                yield
            finally:
                self._semaphore.release()
        else:
            yield


llm_limiter = LLMConcurrencyLimiter()


def init_llm_semaphore(max_concurrent: int) -> None:
    """Initialize the global LLM concurrency semaphore.

    Convenience wrapper around :meth:`LLMConcurrencyLimiter.init`.
    """
    llm_limiter.init(max_concurrent)


@contextmanager
def llm_semaphore() -> Generator[None, None, None]:
    """Context manager that acquires the LLM concurrency semaphore.

    Convenience wrapper around :meth:`LLMConcurrencyLimiter.acquire`.
    """
    with llm_limiter.acquire():
        yield
