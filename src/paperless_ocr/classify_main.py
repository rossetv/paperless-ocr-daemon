"""
Paperless-ngx Classification Daemon
===================================

This script polls Paperless-ngx for documents tagged after OCR and
classifies them using an LLM, applying metadata updates and tags.
"""

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from .classifier import ClassificationProvider
from .classify_worker import ClassificationProcessor, TaxonomyCache
from .config import Settings, setup_libraries
from .logging_config import configure_logging
from .paperless import PaperlessClient


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

    was_idle = False
    try:
        while True:
            try:
                all_docs = list_client.get_documents_by_tag(settings.CLASSIFY_PRE_TAG_ID)
                docs_to_process = [
                    doc
                    for doc in all_docs
                    if not (
                        settings.CLASSIFY_POST_TAG_ID
                        and settings.CLASSIFY_POST_TAG_ID in doc.get("tags", [])
                    )
                ]

                if docs_to_process:
                    was_idle = False
                    taxonomy_cache.refresh()
                    log.info(
                        "Found documents to classify",
                        doc_count=len(docs_to_process),
                        document_workers=settings.DOCUMENT_WORKERS,
                    )

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

                    with ThreadPoolExecutor(
                        max_workers=settings.DOCUMENT_WORKERS
                    ) as executor:
                        future_to_doc = {
                            executor.submit(process_document, doc): doc
                            for doc in docs_to_process
                        }
                        for future in as_completed(future_to_doc):
                            doc = future_to_doc[future]
                            try:
                                future.result()
                            except Exception:
                                log.exception(
                                    "Failed to classify document", doc_id=doc.get("id")
                                )
                else:
                    if not was_idle:
                        log.info("No documents to classify. Waiting...")
                        was_idle = True

                time.sleep(settings.POLL_INTERVAL)
            except KeyboardInterrupt:
                log.info("Ctrl-C received, exiting.")
                break
            except Exception:
                log.exception("An unexpected error occurred in the main loop")
                time.sleep(settings.POLL_INTERVAL)
    finally:
        list_client.close()
        taxonomy_client.close()


if __name__ == "__main__":
    main()
