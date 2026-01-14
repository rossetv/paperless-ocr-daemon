"""
Paperless-ngx AI OCR Daemon
===========================

This script is the main entry point for the Paperless-ngx AI OCR daemon.
It continuously polls a Paperless-ngx instance for documents that have
been tagged for OCR processing. When a new document is found, it is
downloaded, and each page is transcribed by an AI model. The resulting
plain text is then uploaded back to the document, and the tag is updated
to mark it as processed.

The daemon is designed to be resilient, with robust retry mechanisms for
network requests and AI model interactions. It supports multiple OCR
providers and can process multi-page documents in parallel to improve
throughput. The configuration is managed through environment variables,
allowing for flexible deployment in various environments.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from .config import Settings, setup_libraries
from .logging_config import configure_logging
from .ocr import OpenAIProvider
from .paperless import PaperlessClient
from .worker import DocumentProcessor


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
        workers=settings.WORKERS,
        document_workers=settings.DOCUMENT_WORKERS,
        llm_provider=settings.LLM_PROVIDER,
    )

    paperless_client = PaperlessClient(settings)

    was_idle = False
    while True:
        try:
            # Fetch all documents that have the pre-tag
            all_docs = paperless_client.get_documents_to_process()

            # Filter out documents that already have the post-tag
            docs_to_process = [
                doc
                for doc in all_docs
                if settings.POST_TAG_ID not in doc.get("tags", [])
            ]

            if docs_to_process:
                was_idle = False  # Reset idle state
                log.info(
                    "Found documents to process",
                    doc_count=len(docs_to_process),
                    document_workers=settings.DOCUMENT_WORKERS,
                )

                def process_document(doc: dict) -> None:
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
                                "Failed to process document", doc_id=doc.get("id")
                            )
            else:
                if not was_idle:
                    log.info("No documents to process. Waiting...")
                    was_idle = True

            time.sleep(settings.POLL_INTERVAL)
        except KeyboardInterrupt:
            log.info("Ctrl-C received, exiting.")
            break
        except Exception:
            log.exception("An unexpected error occurred in the main loop")
            time.sleep(settings.POLL_INTERVAL)


if __name__ == "__main__":
    main()
