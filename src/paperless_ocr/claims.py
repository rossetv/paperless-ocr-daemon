"""
Processing Tag Claims
=====================

Shared helper for claiming processing tags with a refresh-before and
verify-after workflow to reduce race conditions.
"""

from __future__ import annotations

import structlog

from .paperless import PaperlessClient

log = structlog.get_logger(__name__)


def _extract_tags(doc: dict, *, doc_id: int, context: str) -> set[int]:
    """Return a set of tags from a document payload."""
    tags = doc.get("tags", []) or []
    if not isinstance(tags, list):
        log.warning(
            "Document tags were not a list; treating as empty",
            doc_id=doc_id,
            context=context,
        )
        return set()
    return set(tags)


def claim_processing_tag(
    *,
    paperless_client: PaperlessClient,
    doc_id: int,
    tag_id: int | None,
    purpose: str,
) -> bool:
    """
    Claim a document by adding a processing tag and verifying it persists.
    """
    if not tag_id:
        return True

    try:
        latest = paperless_client.get_document(doc_id)
    except Exception:
        log.exception(
            "Failed to refresh document before claiming processing tag",
            doc_id=doc_id,
            processing_tag_id=tag_id,
            purpose=purpose,
        )
        return False

    current_tags = _extract_tags(latest, doc_id=doc_id, context=f"{purpose}-claim")
    if tag_id in current_tags:
        log.info(
            "Document already claimed",
            doc_id=doc_id,
            processing_tag_id=tag_id,
            purpose=purpose,
        )
        return False

    updated_tags = set(current_tags)
    updated_tags.add(tag_id)
    try:
        paperless_client.update_document_metadata(doc_id, tags=list(updated_tags))
    except Exception:
        log.exception(
            "Failed to claim processing tag",
            doc_id=doc_id,
            processing_tag_id=tag_id,
            purpose=purpose,
        )
        return False

    try:
        verified = paperless_client.get_document(doc_id)
    except Exception:
        log.exception(
            "Failed to refresh document after claiming processing tag",
            doc_id=doc_id,
            processing_tag_id=tag_id,
            purpose=purpose,
        )
        return False

    verified_tags = _extract_tags(
        verified, doc_id=doc_id, context=f"{purpose}-verify"
    )
    if tag_id not in verified_tags:
        log.warning(
            "Processing tag claim could not be verified",
            doc_id=doc_id,
            processing_tag_id=tag_id,
            purpose=purpose,
        )
        return False

    log.info(
        "Claimed document",
        doc_id=doc_id,
        processing_tag_id=tag_id,
        purpose=purpose,
    )
    return True
