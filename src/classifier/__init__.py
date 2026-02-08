"""
Classification domain package.

This package contains:

- the classification provider (prompt + parsing + LLM calls)
- the classification worker that processes a single Paperless document
- the long-running classification daemon entrypoint
"""

from .provider import (
    ClassificationProvider,
    ClassificationResult,
    parse_classification_response,
)
from .worker import ClassificationProcessor, TaxonomyCache

__all__ = [
    "ClassificationProcessor",
    "ClassificationProvider",
    "ClassificationResult",
    "TaxonomyCache",
    "parse_classification_response",
]

