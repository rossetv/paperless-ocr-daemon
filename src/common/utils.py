"""
Text content helpers.

Domain-specific utility functions for detecting redaction markers and
error/refusal content in OCR output.  These are used by both the OCR and
classification pipelines.

- :func:`contains_redacted_marker` — detects ``[REDACTED ...]`` markers.
- :func:`is_error_content` — combines refusal-phrase and redaction detection.

Retry logic has been extracted to :mod:`common.retry`.
"""

from __future__ import annotations

import re
from typing import Iterable

# Pre-compiled at module level for efficiency — matches bracketed text
# containing the word "redacted" in any case, e.g. "[REDACTED]",
# "[NAME REDACTED]", "[REDACTED ADDRESS]".
_REDACTED_RE = re.compile(r"\[[^\]]*redacted[^\]]*\]", re.IGNORECASE)


def contains_redacted_marker(text: str) -> bool:
    """Return ``True`` if *text* includes bracketed redaction markers.

    Examples of detected patterns::

        "[REDACTED]"
        "[REDACTED NAME]"
        "[NAME REDACTED]"

    Args:
        text: The text to search.

    Returns:
        ``True`` if at least one redaction marker is found.
    """
    return bool(_REDACTED_RE.search(text))


def is_error_content(text: str, error_phrases: Iterable[str]) -> bool:
    """Return ``True`` if *text* contains a refusal phrase or redaction marker.

    Used by the classification pipeline to detect OCR output that was stored
    as document content but actually represents a model refusal or error.

    Args:
        text: The document content to check.
        error_phrases: Lower-cased phrases that indicate an error/refusal.

    Returns:
        ``True`` if any refusal phrase or redaction marker is found.
    """
    text_lower = text.lower()
    return contains_redacted_marker(text) or any(
        phrase in text_lower for phrase in error_phrases
    )
