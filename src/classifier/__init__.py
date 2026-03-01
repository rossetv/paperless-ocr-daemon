"""
Classification domain package.

Modules
-------
- **provider** — :class:`ClassificationProvider` (LLM calls + model fallback).
- **result** — :class:`ClassificationResult` dataclass and JSON parser.
- **prompts** — system prompt, JSON schema, and default parameters.
- **worker** — :class:`ClassificationProcessor` orchestrating single-document
  classification.
- **taxonomy** — :class:`TaxonomyCache` for correspondents, types, and tags.
- **tag_filters** — tag deduplication, filtering, and enrichment.
- **content_prep** — OCR content truncation for the classification LLM.
- **metadata** — date, language, and custom-field helpers.
- **normalizers** — string normalization for fuzzy matching.
- **constants** — regex patterns, frozen sets, and phrase lists.
- **daemon** — long-running classification daemon entry point.
"""

from .provider import ClassificationProvider
from .result import ClassificationResult, parse_classification_response
from .taxonomy import TaxonomyCache
from .worker import ClassificationProcessor

__all__ = [
    "ClassificationProcessor",
    "ClassificationProvider",
    "ClassificationResult",
    "TaxonomyCache",
    "parse_classification_response",
]
