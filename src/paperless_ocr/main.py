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

import logging
import time

from .config import Settings, setup_libraries
from .ocr import OpenAIProvider
from .paperless import PaperlessClient
from .worker import DocumentProcessor


def setup_logging() -> None:
    """
    Configures basic logging for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)-3d ▶ %(message)s",
    )


def main() -> None:
    """
    The main loop of the daemon.

    This function continuously polls Paperless-ngx for new documents to process,
    and then hands them off to a `DocumentProcessor` instance.
    """
    setup_logging()
    log = logging.getLogger(__name__)

    try:
        settings = Settings()
        setup_libraries(settings)
    except ValueError as e:
        log.error(f"Configuration error: {e}")
        return

    log.info(
        "Starting daemon (pre=%d post=%d poll=%ds dpi=%d thumb=%dpx workers=%d llm=%s)",
        settings.PRE_TAG_ID,
        settings.POST_TAG_ID,
        settings.POLL_INTERVAL_SECONDS,
        settings.DPI,
        settings.THUMB_SIDE_PX,
        settings.MAX_WORKERS,
        settings.LLM_PROVIDER,
    )

    paperless_client = PaperlessClient(settings)
    ocr_provider = OpenAIProvider(settings)

    while True:
        try:
            docs_to_process = paperless_client.get_documents_to_process()
            for doc in docs_to_process:
                try:
                    processor = DocumentProcessor(
                        doc, paperless_client, ocr_provider, settings
                    )
                    processor.process()
                except Exception as e:
                    log.exception("Failed to process document %s: %s", doc.get("id"), e)
            time.sleep(settings.POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log.info("Ctrl-C received, exiting.")
            break
        except Exception as e:
            log.exception("An unexpected error occurred in the main loop: %s", e)
            time.sleep(settings.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()