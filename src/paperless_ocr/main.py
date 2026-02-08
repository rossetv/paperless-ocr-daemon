"""
Paperless-ngx AI OCR Daemon
===========================

This module is the entry point for the OCR daemon.

High-level behaviour
--------------------

1. Poll Paperless-ngx for documents with the *OCR queue tag* (``PRE_TAG_ID``).
2. For each queued document:
   - Download the original file.
   - Convert the file to images (PDF pages or image frames).
   - Transcribe each page using a vision-capable LLM.
   - Upload the transcription back to Paperless (``content`` field).
   - Update tags to move the document through the pipeline.

The exact tags are configured via environment variables (see ``Settings``).
"""

from __future__ import annotations

from typing import Iterable

import structlog

from .config import Settings, setup_libraries
from .daemon_loop import run_polling_threadpool
from .logging_config import configure_logging
from .ocr import OpenAIProvider
from .paperless import PaperlessClient
from .worker import DocumentProcessor


def _process_document(doc: dict, settings: Settings) -> None:
    """
    Process a single Paperless document.

    The daemon runs multiple documents concurrently; each document gets its own
    Paperless HTTP session. The OCR provider instance is shared across pages
    within the document (pages are processed in parallel).
    """
    paperless = PaperlessClient(settings)
    ocr_provider = OpenAIProvider(settings)
    try:
        processor = DocumentProcessor(
            doc,
            paperless,
            ocr_provider,
            settings,
        )
        processor.process()
    finally:
        paperless.close()

def _iter_docs_to_ocr(list_client: PaperlessClient, settings: Settings) -> Iterable[dict]:
    """
    Yield documents that should be OCR'd.

    Notes on tag hygiene
    --------------------

    Paperless queries here are tag-based (``tags__id=<PRE_TAG_ID>``). If a
    document somehow has both the pre-tag and post-tag, OCR is skipped (to avoid
    repeated OCR cost) and we remove the stale pre-tag so it stops re-appearing
    in the queue.
    """
    log = structlog.get_logger(__name__)
    for doc in list_client.get_documents_by_tag(settings.PRE_TAG_ID):
        doc_id = doc.get("id")
        if not isinstance(doc_id, int):
            log.warning("Skipping document without integer id", doc_id=doc_id)
            continue

        tags_raw = doc.get("tags", []) or []
        tags = set(tags_raw if isinstance(tags_raw, list) else [])

        # If a document has both the "queued" tag and the "processed" tag, treat it
        # as already processed and remove the stale queue tag.
        if settings.POST_TAG_ID in tags:
            if settings.PRE_TAG_ID in tags:
                tags.discard(settings.PRE_TAG_ID)
                try:
                    list_client.update_document_metadata(doc_id, tags=list(tags))
                except Exception:
                    log.exception(
                        "Failed to remove stale pre-tag from already processed document",
                        doc_id=doc_id,
                        pre_tag_id=settings.PRE_TAG_ID,
                        post_tag_id=settings.POST_TAG_ID,
                    )
                else:
                    log.info(
                        "Removed stale pre-tag from already processed document",
                        doc_id=doc_id,
                        pre_tag_id=settings.PRE_TAG_ID,
                        post_tag_id=settings.POST_TAG_ID,
                    )
            continue

        # If configured, the processing tag acts as a best-effort lock. We skip
        # documents that are already claimed to reduce duplicate work.
        if settings.OCR_PROCESSING_TAG_ID and settings.OCR_PROCESSING_TAG_ID in tags:
            continue

        yield doc


def main() -> None:
    """
    The main loop of the daemon.

    This function continuously polls Paperless-ngx for new documents to process,
    and then hands them off to a `DocumentProcessor` instance.
    """
    log = structlog.get_logger(__name__)

    try:
        settings = Settings()
        configure_logging(settings)
        setup_libraries(settings)
    except ValueError as e:
        log.error("Configuration error", error=e)
        return

    log.info(
        "Starting daemon",
        pre_tag_id=settings.PRE_TAG_ID,
        post_tag_id=settings.POST_TAG_ID,
        poll_interval=settings.POLL_INTERVAL,
        ocr_dpi=settings.OCR_DPI,
        ocr_max_side=settings.OCR_MAX_SIDE,
        page_workers=settings.PAGE_WORKERS,
        document_workers=settings.DOCUMENT_WORKERS,
        llm_provider=settings.LLM_PROVIDER,
        ai_models=settings.AI_MODELS,
        ocr_processing_tag_id=settings.OCR_PROCESSING_TAG_ID,
    )

    list_client = PaperlessClient(settings)
    try:
        run_polling_threadpool(
            daemon_name="ocr",
            fetch_work=lambda: list(_iter_docs_to_ocr(list_client, settings)),
            process_item=lambda doc: _process_document(doc, settings),
            poll_interval_seconds=settings.POLL_INTERVAL,
            max_workers=settings.DOCUMENT_WORKERS,
        )
    finally:
        list_client.close()


if __name__ == "__main__":
    main()
