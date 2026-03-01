"""
OCR domain package.

Modules
-------
- **provider** — :class:`OcrProvider` ABC and :class:`OpenAIProvider` implementation.
- **worker** — :class:`DocumentProcessor` orchestrating single-document OCR.
- **daemon** — long-running OCR daemon entry point.
- **prompts** — the system prompt sent to the vision-capable LLM.
"""

from .provider import OcrProvider, OpenAIProvider
from .worker import DocumentProcessor

__all__ = [
    "DocumentProcessor",
    "OcrProvider",
    "OpenAIProvider",
]
