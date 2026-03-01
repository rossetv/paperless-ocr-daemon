"""Extended tests for common.claims — edge cases and failure modes."""

from unittest.mock import MagicMock

from common.claims import claim_processing_tag


def test_claim_returns_true_when_tag_id_none():
    client = MagicMock()
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=None, purpose="test"
    )
    assert result is True
    client.get_document.assert_not_called()


def test_claim_returns_true_when_tag_id_zero():
    client = MagicMock()
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=0, purpose="test"
    )
    assert result is True
    client.get_document.assert_not_called()


def test_claim_returns_false_when_refresh_fails():
    client = MagicMock()
    client.get_document.side_effect = ConnectionError("network error")
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=99, purpose="test"
    )
    assert result is False


def test_claim_returns_false_when_already_claimed():
    client = MagicMock()
    client.get_document.return_value = {"id": 1, "tags": [10, 99]}
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=99, purpose="test"
    )
    assert result is False
    client.update_document_metadata.assert_not_called()


def test_claim_returns_false_when_patch_fails():
    client = MagicMock()
    client.get_document.return_value = {"id": 1, "tags": [10]}
    client.update_document_metadata.side_effect = ConnectionError("fail")
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=99, purpose="test"
    )
    assert result is False


def test_claim_returns_false_when_verify_fails():
    client = MagicMock()
    client.get_document.side_effect = [
        {"id": 1, "tags": [10]},          # Step 1: refresh
        ConnectionError("verify fail"),    # Step 4: verify
    ]
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=99, purpose="test"
    )
    assert result is False


def test_claim_returns_false_when_verify_shows_tag_missing():
    client = MagicMock()
    client.get_document.side_effect = [
        {"id": 1, "tags": [10]},         # Step 1: refresh (no claim tag)
        {"id": 1, "tags": [10]},         # Step 4: verify (tag was reverted)
    ]
    result = claim_processing_tag(
        paperless_client=client, doc_id=1, tag_id=99, purpose="test"
    )
    assert result is False
