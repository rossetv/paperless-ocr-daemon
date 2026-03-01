"""
String Normalizers
==================

Small, pure functions for normalizing organisation names and general strings.
Used by the taxonomy cache (matching existing Paperless items) and tag filters
(removing tags that duplicate the correspondent or document type).

These live in their own module so both ``taxonomy`` and ``tag_filters`` can
import them without creating a circular dependency.
"""

from __future__ import annotations

import re

# Common corporate suffixes stripped when comparing organisation names.
# Kept as a set for O(1) membership tests.
COMPANY_SUFFIXES: frozenset[str] = frozenset({
    "ab",
    "as",
    "bv",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "oy",
    "plc",
    "sa",
    "sarl",
    "spa",
})


def normalize_simple(value: str) -> str:
    """
    Collapse whitespace and lowercase a string.

    >>> normalize_simple("  Bank  Statement ")
    'bank statement'
    """
    return " ".join(value.lower().split())


def normalize_name(value: str) -> str:
    """
    Normalize an organisation name for fuzzy matching.

    Strips punctuation, lowercases, and removes trailing corporate suffixes
    so that *"Revolut Ltd."* and *"Revolut"* compare as equal.

    >>> normalize_name("Revolut Ltd.")
    'revolut'
    """
    cleaned = re.sub(r"[^a-z0-9\s]", "", value.lower())
    parts = cleaned.split()
    while parts and parts[-1] in COMPANY_SUFFIXES:
        parts.pop()
    return " ".join(parts)
