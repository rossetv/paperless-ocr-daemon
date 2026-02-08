"""
Paperless-ngx Classification Daemon
===================================

This script polls Paperless-ngx for documents tagged after OCR and
classifies them using an LLM, applying metadata updates and tags.
"""

from __future__ import annotations

from typing import Iterable

import structlog

from .classifier import ClassificationProvider
from .classify_worker import ClassificationProcessor, TaxonomyCache
from .config import Settings, setup_libraries
from .daemon_loop import run_polling_threadpool
from .logging_config import configure_logging
from .paperless import PaperlessClient


def _iter_docs_to_classify(
    list_client: PaperlessClient, settings: Settings
) -> Iterable[dict]:
    """
    Yield documents that should be classified.

    If ``CLASSIFY_POST_TAG_ID`` is configured and a document already has it,
    classification is skipped and we remove the stale pre-tag so the document
    stops re-appearing in the queue.
    """
    log = structlog.get_logger(__name__)
    for doc in list_client.get_documents_by_tag(settings.CLASSIFY_PRE_TAG_ID):
        doc_id = doc.get("id")
        if not isinstance(doc_id, int):
            log.warning("Skipping document without integer id", doc_id=doc_id)
            continue

        tags_raw = doc.get("tags", []) or []
        tags = set(tags_raw if isinstance(tags_raw, list) else [])

        if settings.CLASSIFY_POST_TAG_ID and settings.CLASSIFY_POST_TAG_ID in tags:
            if settings.CLASSIFY_PRE_TAG_ID in tags:
                tags.discard(settings.CLASSIFY_PRE_TAG_ID)
                if settings.CLASSIFY_PROCESSING_TAG_ID:
                    tags.discard(settings.CLASSIFY_PROCESSING_TAG_ID)
                try:
                    list_client.update_document_metadata(doc_id, tags=list(tags))
                except Exception:
                    log.exception(
                        "Failed to remove stale classify-pre tag from already processed document",
                        doc_id=doc_id,
                        classify_pre_tag_id=settings.CLASSIFY_PRE_TAG_ID,
                        classify_post_tag_id=settings.CLASSIFY_POST_TAG_ID,
                    )
                else:
                    log.info(
                        "Removed stale classify-pre tag from already processed document",
                        doc_id=doc_id,
                        classify_pre_tag_id=settings.CLASSIFY_PRE_TAG_ID,
                        classify_post_tag_id=settings.CLASSIFY_POST_TAG_ID,
                    )
            continue

        if (
            settings.CLASSIFY_PROCESSING_TAG_ID
            and settings.CLASSIFY_PROCESSING_TAG_ID in tags
        ):
            continue

        yield doc


def main() -> None:
    """Main loop for the classification daemon."""
    log = structlog.get_logger(__name__)

    try:
        settings = Settings()
        configure_logging(settings)
        setup_libraries(settings)
    except ValueError as e:
        log.error("Configuration error", error=e)
        return

    log.info(
        "Starting classification daemon",
        classify_pre_tag_id=settings.CLASSIFY_PRE_TAG_ID,
        classify_post_tag_id=settings.CLASSIFY_POST_TAG_ID,
        poll_interval=settings.POLL_INTERVAL,
        document_workers=settings.DOCUMENT_WORKERS,
        llm_provider=settings.LLM_PROVIDER,
        ai_models=settings.AI_MODELS,
        classify_processing_tag_id=settings.CLASSIFY_PROCESSING_TAG_ID,
    )

    list_client = PaperlessClient(settings)
    taxonomy_client = PaperlessClient(settings)
    taxonomy_cache = TaxonomyCache(taxonomy_client, settings.CLASSIFY_TAXONOMY_LIMIT)

    def process_document(doc: dict) -> None:
        paperless = PaperlessClient(settings)
        classifier = ClassificationProvider(settings)
        try:
            processor = ClassificationProcessor(
                doc,
                paperless,
                classifier,
                taxonomy_cache,
                settings,
            )
            processor.process()
        finally:
            paperless.close()

    try:
        run_polling_threadpool(
            daemon_name="classifier",
            fetch_work=lambda: list(_iter_docs_to_classify(list_client, settings)),
            process_item=process_document,
            before_each_batch=lambda _: taxonomy_cache.refresh(),
            poll_interval_seconds=settings.POLL_INTERVAL,
            max_workers=settings.DOCUMENT_WORKERS,
        )
    finally:
        list_client.close()
        taxonomy_client.close()


if __name__ == "__main__":
    main()
