"""
Processing Tag Claims
=====================

Shared helper for claiming a processing-lock tag with a refresh-before and
verify-after workflow to reduce race conditions in concurrent environments.

When multiple daemon instances (or threads) process the same Paperless queue,
they may both pick up the same document.  The processing tag acts as a
best-effort lock: the first worker to successfully add the tag and verify that
it persists "owns" the document.
"""

from __future__ import annotations

import structlog

from .paperless import PaperlessClient
from .tags import extract_tags

log = structlog.get_logger(__name__)


def claim_processing_tag(
    *,
    paperless_client: PaperlessClient,
    doc_id: int,
    tag_id: int | None,
    purpose: str,
) -> bool:
    """
    Claim a document by adding a processing-lock tag and verifying it persists.

    The workflow is:

    1. **Refresh** — fetch the document to get the latest tags.
    2. **Check** — if the tag is already present, another worker claimed it.
    3. **Patch** — add the tag.
    4. **Verify** — re-fetch and confirm the tag is still there (Paperless may
       have reverted it if two workers patched simultaneously).

    Returns ``True`` when the claim succeeds, ``False`` otherwise.  When
    *tag_id* is ``None`` or ``0`` (lock not configured), returns ``True``
    immediately.
    """
    if not tag_id:
        return True

    # Step 1: Refresh
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

    # Step 2: Check
    current_tags = extract_tags(latest, doc_id=doc_id, context=f"{purpose}-claim")
    if tag_id in current_tags:
        log.info(
            "Document already claimed",
            doc_id=doc_id,
            processing_tag_id=tag_id,
            purpose=purpose,
        )
        return False

    # Step 3: Patch
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

    # Step 4: Verify
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

    verified_tags = extract_tags(verified, doc_id=doc_id, context=f"{purpose}-verify")
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
