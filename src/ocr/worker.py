"""
Document Processing Worker
==========================

Orchestrates the end-to-end OCR processing of a single Paperless document:

1. Download the original file from Paperless.
2. Convert it to images (PDF pages or image frames).
3. Transcribe each page in parallel using a vision-capable LLM.
4. Assemble the transcriptions into a single text with page headers.
5. Upload the text and update tags in Paperless.

The :class:`DocumentProcessor` is instantiated per-document by the daemon's
thread pool.  Each instance gets its own :class:`PaperlessClient` and
:class:`OcrProvider`.
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
from common.tags import clean_pipeline_tags, get_latest_tags, release_processing_tag
from common.utils import contains_redacted_marker
from .provider import OcrProvider

log = structlog.get_logger(__name__)

# Inserted into the document content when OCR fails for a page, so downstream
# steps (and humans) can see where the pipeline broke.
OCR_ERROR_MARKER = "[OCR ERROR]"


class DocumentProcessor:
    """
    Orchestrates the OCR processing of a single Paperless document.

    Instantiated per-document by the daemon's thread pool.
    """

    def __init__(
        self,
        doc: dict,
        paperless_client: PaperlessClient,
        ocr_provider: OcrProvider,
        settings: Settings,
    ):
        self.doc = doc
        self.paperless_client = paperless_client
        self.ocr_provider = ocr_provider
        self.settings = settings
        self.doc_id: int = doc["id"]
        self.title: str = doc.get("title") or "<untitled>"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self) -> None:
        """
        Execute the end-to-end OCR workflow for this document.

        Steps: refresh → check error tag → claim lock → download →
        convert to images → OCR pages → assemble text → update Paperless →
        release lock.
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
                log.warning("Document has error tag; skipping OCR", doc_id=self.doc_id)
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
                release_processing_tag(
                    self.paperless_client,
                    self.doc_id,
                    self.settings.OCR_PROCESSING_TAG_ID,
                    purpose="ocr",
                )
            self._log_ocr_stats()

        elapsed = (dt.datetime.now() - start_time).total_seconds()
        log.info(
            "Finished processing document",
            doc_id=self.doc_id,
            elapsed_time=f"{elapsed:.2f}s",
        )

    # ------------------------------------------------------------------
    # Document refresh
    # ------------------------------------------------------------------

    def _refresh_document(self) -> dict | None:
        """Refresh the document from Paperless and update the local cache."""
        try:
            latest = self.paperless_client.get_document(self.doc_id)
        except Exception:
            log.exception("Failed to refresh document metadata", doc_id=self.doc_id)
            return None
        self.doc = latest
        return latest

    # ------------------------------------------------------------------
    # Processing-lock tag
    # ------------------------------------------------------------------

    def _claim_processing_tag(self) -> bool:
        """Claim the OCR processing-lock tag with refresh-before/verify-after."""
        return claim_processing_tag(
            paperless_client=self.paperless_client,
            doc_id=self.doc_id,
            tag_id=self.settings.OCR_PROCESSING_TAG_ID,
            purpose="ocr",
        )

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------

    def _bytes_to_images(self, content: bytes, content_type: str) -> list[Image.Image]:
        """
        Convert downloaded bytes into a list of PIL Images.

        - PDFs are rasterised into one image per page.
        - Image formats (PNG/JPEG/TIFF/…) are loaded via Pillow.
        - Multi-frame images (e.g. TIFF) are expanded into one image per frame.

        The returned images are fully loaded into memory so they do not depend
        on any open file handles.
        """
        if "pdf" in content_type.lower():
            return convert_from_bytes(content, dpi=self.settings.OCR_DPI)
        try:
            img = Image.open(BytesIO(content))
            img.load()
            if getattr(img, "n_frames", 1) > 1:
                frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
                img.close()
                return frames
            single = img.copy()
            img.close()
            return [single]
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Unable to open image: {e}") from e

    # ------------------------------------------------------------------
    # Parallel OCR
    # ------------------------------------------------------------------

    def _ocr_pages_in_parallel(
        self, images: list[Image.Image]
    ) -> tuple[list[tuple[str, str]], list[int]]:
        """
        Run OCR on each page concurrently and preserve the original order.

        Returns ``(page_results, failed_page_numbers)`` where each page result
        is a ``(text, model_used)`` tuple.
        """
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
            results: list[tuple[str, str]] = [("", "")] * len(images)
            failed_pages: list[int] = []
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception:
                    log.exception("OCR failed on page", page_num=index + 1)
                    failed_pages.append(index + 1)
                    results[index] = (
                        f"{OCR_ERROR_MARKER} Failed to OCR page {index + 1}.",
                        "",
                    )
            return results, failed_pages

    # ------------------------------------------------------------------
    # Text assembly
    # ------------------------------------------------------------------

    def _assemble_full_text(
        self, images: list[Image.Image], page_results: list[tuple[str, str]]
    ) -> tuple[str, set[str]]:
        """
        Combine per-page OCR results into a single document text.

        Multi-page documents get ``--- Page N ---`` headers.  A footer listing
        all models used is appended at the end.
        """
        sections: list[str] = []
        models_used: set[str] = set()
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

    # ------------------------------------------------------------------
    # Paperless update
    # ------------------------------------------------------------------

    def _update_paperless_document(self, full_text: str, models_used: set[str]) -> None:
        """
        Upload OCR text and update tags in Paperless.

        Detects error conditions (empty text, refusal markers, OCR errors)
        and routes to :meth:`_finalize_with_error` instead of the happy path.
        """
        if not full_text.strip():
            log.warning("OCR produced no text; marking error", doc_id=self.doc_id)
            self._finalize_with_error(
                get_latest_tags(
                    self.paperless_client, self.doc_id, fallback_doc=self.doc
                ),
                content=full_text,
            )
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
            self._finalize_with_error(
                get_latest_tags(
                    self.paperless_client, self.doc_id, fallback_doc=self.doc
                ),
                content=full_text,
            )
            return

        # Happy path: swap the pre-tag for the post-tag
        current_tags = get_latest_tags(
            self.paperless_client, self.doc_id, fallback_doc=self.doc
        )
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

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _finalize_with_error(self, tags: set[int], *, content: str | None = None) -> None:
        """
        Mark the document as errored and remove all pipeline tags.

        Preserves user-assigned tags and adds ``ERROR_TAG_ID`` when configured.
        If *content* is provided, the Paperless ``content`` field is also
        updated (useful when the OCR output contains refusal markers that
        should be visible to human reviewers).
        """
        updated = clean_pipeline_tags(tags, self.settings)
        if self.settings.ERROR_TAG_ID:
            updated.add(self.settings.ERROR_TAG_ID)

        if content is None:
            self.paperless_client.update_document_metadata(self.doc_id, tags=list(updated))
        else:
            self.paperless_client.update_document(self.doc_id, content, list(updated))

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _log_ocr_stats(self) -> None:
        """Log OCR model stats if the provider exposes them."""
        if not hasattr(self.ocr_provider, "get_stats"):
            return
        stats = self.ocr_provider.get_stats()
        if not stats or not stats.get("attempts"):
            return
        log.info("OCR stats", doc_id=self.doc_id, **stats)
