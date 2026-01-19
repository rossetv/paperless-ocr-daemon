"""
Document Processing Worker
==========================

This module defines the `DocumentProcessor` class, which is responsible for
orchestrating the end-to-end processing of a single document from
Paperless-ngx. It brings together the functionality from the other modules,
including the Paperless client, the OCR provider, and image conversion
utilities.

The `DocumentProcessor` handles the entire lifecycle of a document's OCR
process: downloading the file, converting it to images, sending each page
to the OCR provider, assembling the transcribed text, and finally, updating
the document in Paperless-ngx with the new content and tags. This class is
designed to be instantiated for each document that needs processing.
"""

import datetime as dt
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import IO, List

import structlog
from PIL import Image, ImageSequence, UnidentifiedImageError
from pdf2image import convert_from_path

from .config import Settings
from .ocr import OcrProvider
from .paperless import PaperlessClient

log = structlog.get_logger(__name__)


class DocumentProcessor:
    """
    Orchestrates the processing of a single Paperless document.
    """

    def __init__(
        self,
        doc: dict,
        paperless_client: PaperlessClient,
        ocr_provider: OcrProvider,
        settings: Settings,
    ):
        """
        Initializes the processor with the document, clients, and settings.
        """
        self.doc = doc
        self.paperless_client = paperless_client
        self.ocr_provider = ocr_provider
        self.settings = settings
        self.doc_id = doc["id"]
        self.title = doc.get("title") or "<untitled>"

    def process(self) -> None:
        """
        Execute the end-to-end processing of the document.
        """
        log.info("Processing document", doc_id=self.doc_id, title=self.title)
        start_time = dt.datetime.now()
        claimed = False
        try:
            latest = self._refresh_document()
            if latest is None:
                return
            current_tags = set(latest.get("tags", []) or [])

            if self.settings.ERROR_TAG_ID and self.settings.ERROR_TAG_ID in current_tags:
                log.warning(
                    "Document has error tag; skipping OCR",
                    doc_id=self.doc_id,
                )
                self._set_error_only_tags()
                return

            claimed = self._claim_processing_tag(current_tags)
            if not claimed:
                return

            content, content_type = self.paperless_client.download_content(self.doc_id)
            extension = ".pdf" if "pdf" in content_type else ".bin"

            with tempfile.NamedTemporaryFile(delete=True, suffix=extension) as temp_file:
                temp_file.write(content)
                temp_file.flush()  # Ensure content is written to disk

                images = self._file_to_images(temp_file.name, content_type)

            if not images:
                log.warning("Document has no pages to process", doc_id=self.doc_id)
                return

            page_results = self._ocr_pages_in_parallel(images)

            full_text, models_used = self._assemble_full_text(images, page_results)

            self._update_paperless_document(full_text, models_used)
        finally:
            if claimed:
                self._release_processing_tag()
            self._log_ocr_stats()

        elapsed_time = (dt.datetime.now() - start_time).total_seconds()
        log.info(
            "Finished processing document",
            doc_id=self.doc_id,
            elapsed_time=f"{elapsed_time:.2f}s",
        )

    def _refresh_document(self) -> dict | None:
        """Refresh the document from Paperless and update the local cache."""
        try:
            latest = self.paperless_client.get_document(self.doc_id)
        except Exception:
            log.exception("Failed to refresh document metadata", doc_id=self.doc_id)
            return None
        self.doc = latest
        return latest

    def _claim_processing_tag(self, current_tags: set[int]) -> bool:
        """Add the OCR processing tag if configured; return True if claimed."""
        tag_id = self.settings.OCR_PROCESSING_TAG_ID
        if not tag_id:
            return True
        if tag_id in current_tags:
            log.info(
                "Document already claimed for OCR",
                doc_id=self.doc_id,
                processing_tag_id=tag_id,
            )
            return False
        updated_tags = set(current_tags)
        updated_tags.add(tag_id)
        try:
            self.paperless_client.update_document_metadata(
                self.doc_id, tags=list(updated_tags)
            )
        except Exception:
            log.exception(
                "Failed to claim OCR processing tag",
                doc_id=self.doc_id,
                processing_tag_id=tag_id,
            )
            return False
        log.info(
            "Claimed document for OCR",
            doc_id=self.doc_id,
            processing_tag_id=tag_id,
        )
        return True

    def _release_processing_tag(self) -> None:
        """Remove the OCR processing tag if it is still present."""
        tag_id = self.settings.OCR_PROCESSING_TAG_ID
        if not tag_id:
            return
        try:
            latest = self.paperless_client.get_document(self.doc_id)
        except Exception:
            log.exception(
                "Failed to refresh document before releasing OCR tag", doc_id=self.doc_id
            )
            return
        tags = set(latest.get("tags", []) or [])
        if tag_id not in tags:
            return
        tags.discard(tag_id)
        try:
            self.paperless_client.update_document_metadata(self.doc_id, tags=list(tags))
        except Exception:
            log.exception(
                "Failed to release OCR processing tag",
                doc_id=self.doc_id,
                processing_tag_id=tag_id,
            )

    def _log_ocr_stats(self) -> None:
        """Log OCR model stats if the provider exposes them."""
        if not hasattr(self.ocr_provider, "get_stats"):
            return
        stats = self.ocr_provider.get_stats()
        if not stats or not stats.get("attempts"):
            return
        log.info("OCR stats", doc_id=self.doc_id, **stats)

    def _file_to_images(self, path: str, content_type: str) -> list[Image.Image]:
        """Convert the downloaded file to a list of PIL.Image instances."""
        if "pdf" in content_type:
            return convert_from_path(path, dpi=self.settings.OCR_DPI)
        try:
            img = Image.open(path)
            img.load()
            if getattr(img, "n_frames", 1) > 1:
                frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
                img.close()
                return frames
            return [img]
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Unable to open image: {e}") from e

    def _ocr_pages_in_parallel(
        self, images: list[Image.Image]
    ) -> list[tuple[str, str]]:
        """Run OCR on each page concurrently and preserve the original order."""
        with ThreadPoolExecutor(max_workers=self.settings.PAGE_WORKERS) as executor:
            future_to_index = {
                executor.submit(self.ocr_provider.transcribe_image, img): i
                for i, img in enumerate(images)
            }
            results = [("", "")] * len(images)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception:
                    log.exception("OCR failed on page", page_num=index + 1)
            return results

    def _assemble_full_text(
        self, images: list[Image.Image], page_results: list[tuple[str, str]]
    ) -> tuple[str, set[str]]:
        """
        Combine the OCR results from all pages into a single string.
        """
        sections = []
        models_used = set()
        for i, (text, model) in enumerate(page_results, 1):
            if text.strip():
                header = f"--- Page {i} ---\n" if len(images) > 1 else ""
                sections.append(f"{header}{text}")
                if model:
                    models_used.add(model)

        full_text = "\n\n".join(sections)
        if models_used:
            footer = f"Transcribed by model: {', '.join(sorted(models_used))}"
            full_text = f"{full_text}\n\n{footer}" if full_text else footer
        return full_text, models_used

    def _update_paperless_document(self, full_text: str, models_used: set[str]) -> None:
        """
        Update the document in Paperless with the new content and tags.
        """
        if self.settings.ERROR_TAG_ID and (
            self.settings.REFUSAL_MARK in full_text or "[REDACTED]" in full_text
        ):
            self.paperless_client.update_document(
                self.doc_id, full_text, [self.settings.ERROR_TAG_ID]
            )
            log.warning(
                "OCR produced refusal/redacted content; marking error",
                doc_id=self.doc_id,
            )
            return

        current_tags = self._get_latest_tags()
        current_tags.discard(self.settings.PRE_TAG_ID)
        if self.settings.OCR_PROCESSING_TAG_ID:
            current_tags.discard(self.settings.OCR_PROCESSING_TAG_ID)
        current_tags.add(self.settings.POST_TAG_ID)

        self.paperless_client.update_document(
            self.doc_id, full_text, list(current_tags)
        )
        log.info(
            "Updated document tags",
            doc_id=self.doc_id,
            removed_tag=self.settings.PRE_TAG_ID,
            added_tag=self.settings.POST_TAG_ID,
        )

    def _get_latest_tags(self) -> set[int]:
        """Return the latest document tags, falling back to cached tags."""
        cached_tags = self.doc.get("tags", []) or []
        try:
            latest = self.paperless_client.get_document(self.doc_id)
        except Exception:
            log.exception(
                "Failed to refresh document before updating tags; using cached tags",
                doc_id=self.doc_id,
            )
            return set(cached_tags)
        latest_tags = latest.get("tags", [])
        if not isinstance(latest_tags, list):
            log.warning(
                "Document tags were not a list; using cached tags",
                doc_id=self.doc_id,
            )
            return set(cached_tags)
        return set(latest_tags)

    def _set_error_only_tags(self) -> None:
        """Replace all tags with the error tag."""
        if not self.settings.ERROR_TAG_ID:
            return
        self.paperless_client.update_document_metadata(
            self.doc_id, tags=[self.settings.ERROR_TAG_ID]
        )
