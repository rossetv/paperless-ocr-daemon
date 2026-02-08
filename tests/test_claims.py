from unittest.mock import MagicMock

from common.claims import claim_processing_tag


def test_claim_processing_tag_verifies_after_patch():
    client = MagicMock()
    client.get_document.side_effect = [
        {"id": 1, "tags": [10]},
        {"id": 1, "tags": [10, 99]},
    ]

    claimed = claim_processing_tag(
        paperless_client=client,
        doc_id=1,
        tag_id=99,
        purpose="ocr",
    )

    assert claimed is True
    args, kwargs = client.update_document_metadata.call_args
    assert args[0] == 1
    assert set(kwargs["tags"]) == {10, 99}


def test_claim_processing_tag_fails_when_tags_remain_stale():
    client = MagicMock()
    client.get_document.side_effect = [
        {"id": 1, "tags": [10]},
        {"id": 1, "tags": [10]},
    ]

    claimed = claim_processing_tag(
        paperless_client=client,
        doc_id=1,
        tag_id=99,
        purpose="ocr",
    )

    assert claimed is False
    args, kwargs = client.update_document_metadata.call_args
    assert args[0] == 1
    assert set(kwargs["tags"]) == {10, 99}
