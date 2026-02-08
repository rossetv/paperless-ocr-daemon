"""
OCR domain package.

This package contains:

- the OCR provider abstraction and implementations
- the OCR worker that processes a single Paperless document
- the long-running OCR daemon entrypoint
"""

from .provider import OcrProvider, OpenAIProvider
from .worker import DocumentProcessor

__all__ = [
    "DocumentProcessor",
    "OcrProvider",
    "OpenAIProvider",
]

