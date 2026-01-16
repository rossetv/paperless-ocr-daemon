"""
Shared LLM helpers.

This module centralizes the OpenAI-compatible chat completion call so OCR and
classification reuse the same retry behavior and logging patterns.
"""

import openai

from .utils import retry


class OpenAIChatMixin:
    """
    Mixin providing a retried OpenAI-compatible chat completion call.

    The mixin expects ``self.settings`` to expose ``MAX_RETRIES`` and
    ``MAX_RETRY_BACKOFF_SECONDS`` for the retry decorator.
    """

    @retry(retryable_exceptions=(openai.APIError,))
    def _create_completion(self, **kwargs):
        """Call the OpenAI-compatible chat completion API with retries."""
        return openai.chat.completions.create(**kwargs)
