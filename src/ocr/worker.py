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

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import structlog
from PIL import Image, ImageSequence, UnidentifiedImageError
from pdf2image import convert_from_bytes

from common.claims import claim_processing_tag
from common.config import Settings
from common.paperless import PaperlessClient
from common.utils import contains_redacted_marker
from .provider import OcrProvider

log = structlog.get_logger(__name__)

OCR_ERROR_MARKER = "[OCR ERROR]"


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
                self._finalize_with_error(current_tags)
                return

            claimed = self._claim_processing_tag()
            if not claimed:
                return

            content, content_type = self.paperless_client.download_content(self.doc_id)
            images = self._bytes_to_images(content, content_type)

            if not images:
                log.warning("Document has no pages to process", doc_id=self.doc_id)
                return

            try:
                page_results, failed_pages = self._ocr_pages_in_parallel(images)
            finally:
                for image in images:
                    try:
                        image.close()
                    except Exception:
                        log.warning("Failed to close image", doc_id=self.doc_id)

            if failed_pages:
                log.warning(
                    "OCR failed on some pages; marking document as error",
                    doc_id=self.doc_id,
                    failed_pages=failed_pages,
                )

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

    def _claim_processing_tag(self) -> bool:
        """Claim the OCR processing tag with verification."""
        return claim_processing_tag(
            paperless_client=self.paperless_client,
            doc_id=self.doc_id,
            tag_id=self.settings.OCR_PROCESSING_TAG_ID,
            purpose="ocr",
        )

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

    def _bytes_to_images(self, content: bytes, content_type: str) -> list[Image.Image]:
        """
        Convert downloaded bytes into a list of PIL Images.

        - PDFs are rasterised into one image per page.
        - Image formats (PNG/JPEG/TIFF/...) are loaded via Pillow.
        - Multi-frame images (e.g. TIFF) are expanded into one image per frame.

        The returned images are fully loaded into memory, so they do not depend on
        any open file handles.
        """
        if "pdf" in content_type.lower():
            return convert_from_bytes(content, dpi=self.settings.OCR_DPI)
        try:
            img = Image.open(BytesIO(content))
            img.load()  # fully decode so the underlying buffer can be released
            if getattr(img, "n_frames", 1) > 1:
                frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
                img.close()
                return frames
            single = img.copy()
            img.close()
            return [single]
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Unable to open image: {e}") from e

    def _ocr_pages_in_parallel(
        self, images: list[Image.Image]
    ) -> tuple[list[tuple[str, str]], list[int]]:
        """Run OCR on each page concurrently and preserve the original order."""
        with ThreadPoolExecutor(max_workers=self.settings.PAGE_WORKERS) as executor:
            future_to_index = {
                executor.submit(
                    self.ocr_provider.transcribe_image,
                    img,
                    doc_id=self.doc_id,
                    page_num=i + 1,
                ): i
                for i, img in enumerate(images)
            }
            results = [("", "")] * len(images)
            failed_pages = []
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception:
                    log.exception("OCR failed on page", page_num=index + 1)
                    failed_pages.append(index + 1)
                    # Insert a marker into the document content so downstream
                    # steps (and humans) can see this was a pipeline failure.
                    results[index] = (
                        f"{OCR_ERROR_MARKER} Failed to OCR page {index + 1}.",
                        "",
                    )
            return results, failed_pages

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
                header = ""
                if len(images) > 1:
                    header = f"--- Page {i}"
                    if self.settings.OCR_INCLUDE_PAGE_MODELS and model:
                        header += f" ({model})"
                    header += " ---\n"
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
        if not full_text.strip():
            # If OCR produced nothing, passing an empty "content" field into the
            # classification daemon causes a requeue loop. Treat this as an OCR
            # failure and mark the document for manual review.
            log.warning("OCR produced no text; marking error", doc_id=self.doc_id)
            self._finalize_with_error(self._get_latest_tags(), content=full_text)
            return

        if (
            OCR_ERROR_MARKER in full_text
            or self.settings.REFUSAL_MARK in full_text
            or contains_redacted_marker(full_text)
        ):
            log.warning(
                "OCR produced error/refusal/redacted markers; marking error",
                doc_id=self.doc_id,
            )
            self._finalize_with_error(self._get_latest_tags(), content=full_text)
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

    def _finalize_with_error(self, tags: set[int], *, content: str | None = None) -> None:
        """
        Mark the document as errored and remove pipeline tags.

        Tag behaviour is intentionally conservative: we remove only the tags used
        by this pipeline, keep any user-assigned tags, and add ``ERROR_TAG_ID``
        when configured.

        If ``content`` is provided, we update the Paperless ``content`` field as
        well (useful when the OCR output contains refusal markers).
        """
        updated = set(tags)

        # Remove pipeline tags so the document does not get re-queued indefinitely.
        for tag_id in (
            self.settings.PRE_TAG_ID,
            self.settings.POST_TAG_ID,
            self.settings.CLASSIFY_PRE_TAG_ID,
        ):
            updated.discard(tag_id)
        if self.settings.CLASSIFY_POST_TAG_ID:
            updated.discard(self.settings.CLASSIFY_POST_TAG_ID)
        if self.settings.OCR_PROCESSING_TAG_ID:
            updated.discard(self.settings.OCR_PROCESSING_TAG_ID)
        if self.settings.CLASSIFY_PROCESSING_TAG_ID:
            updated.discard(self.settings.CLASSIFY_PROCESSING_TAG_ID)

        if self.settings.ERROR_TAG_ID:
            updated.add(self.settings.ERROR_TAG_ID)

        if content is None:
            self.paperless_client.update_document_metadata(self.doc_id, tags=list(updated))
        else:
            self.paperless_client.update_document(self.doc_id, content, list(updated))
