"""Tests for classifier.normalizers — string normalization."""

from classifier.normalizers import COMPANY_SUFFIXES, normalize_name, normalize_simple


# ---------------------------------------------------------------------------
# normalize_simple
# ---------------------------------------------------------------------------


def test_normalize_simple_basic():
    assert normalize_simple("  Bank  Statement ") == "bank statement"


def test_normalize_simple_empty():
    assert normalize_simple("") == ""


def test_normalize_simple_mixed_case():
    assert normalize_simple("InVoIcE") == "invoice"


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


def test_normalize_name_strips_company_suffix():
    assert normalize_name("Revolut Ltd.") == "revolut"


def test_normalize_name_strips_multiple_suffixes():
    assert normalize_name("Foo Corp Ltd") == "foo"


def test_normalize_name_strips_punctuation():
    assert normalize_name("O'Reilly & Co.") == "oreilly"


def test_normalize_name_no_suffix():
    assert normalize_name("Amazon") == "amazon"


def test_normalize_name_all_suffixes_returns_empty():
    assert normalize_name("Ltd") == ""


def test_normalize_name_empty():
    assert normalize_name("") == ""


# ---------------------------------------------------------------------------
# COMPANY_SUFFIXES
# ---------------------------------------------------------------------------


def test_company_suffixes_is_frozenset():
    assert isinstance(COMPANY_SUFFIXES, frozenset)
    assert "ltd" in COMPANY_SUFFIXES
    assert "gmbh" in COMPANY_SUFFIXES
    assert "inc" in COMPANY_SUFFIXES
