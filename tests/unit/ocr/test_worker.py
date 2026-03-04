"""
Comprehensive unit tests for ocr.worker.DocumentProcessor.

Tests cover:
- process() happy path: full pipeline success
- process() claim fails -> returns early
- process() error tag present -> skips with finalize_with_error
- process() refresh failure -> returns
- process() download failure -> releases lock
- process() image conversion failure -> finalizes with error
- process() always releases processing lock in finally
- _bytes_to_images delegates to image_converter
- _ocr_pages_in_parallel: handles errors, preserves order
- _assemble_full_text: delegates to text_assembly
- _update_paperless_document: marks error on refusal/empty/redacted
- _update_paperless_document: happy path swaps tags
- _finalize_with_error: adds error tag, cleans pipeline tags
- _finalize_with_error: no error tag configured -> just cleans tags
- _finalize_with_error: with content updates content
- _log_ocr_stats: zero stats, normal stats, no get_stats method
- Images always closed (even on error)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from ocr.worker import DocumentProcessor
from ocr.text_assembly import OCR_ERROR_MARKER


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_settings(**overrides):
    from tests.helpers.factories import make_settings_obj
    return make_settings_obj(**overrides)


def _make_mock_paperless(**overrides):
    from tests.helpers.mocks import make_mock_paperless
    return make_mock_paperless(**overrides)


def _make_mock_ocr_provider(**overrides):
    from tests.helpers.mocks import make_mock_ocr_provider
    return make_mock_ocr_provider(**overrides)


def _make_processor(
    doc=None,
    paperless=None,
    ocr_provider=None,
    settings=None,
    **setting_overrides,
):
    """Create a DocumentProcessor with mocked dependencies."""
    if doc is None:
        from tests.helpers.factories import make_document
        doc = make_document()
    if settings is None:
        settings = _make_settings(**setting_overrides)
    if paperless is None:
        paperless = _make_mock_paperless()
    if ocr_provider is None:
        ocr_provider = _make_mock_ocr_provider()
    return DocumentProcessor(doc, paperless, ocr_provider, settings)


def _make_image():
    """Create a simple test image."""
    return Image.new("RGB", (10, 10), color="red")


# -----------------------------------------------------------------------
# process() — happy path
# -----------------------------------------------------------------------

class TestProcessHappyPath:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images")
    @patch("ocr.worker.assemble_full_text")
    def test_full_pipeline_success(
        self, mock_assemble, mock_b2i, mock_claim, mock_release
    ):
        # Arrange
        settings = _make_settings(
            OCR_PROCESSING_TAG_ID=999,
            ERROR_TAG_ID=552,
        )
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {
            "id": 1, "title": "Test", "tags": [443]
        }
        paperless.download_content.return_value = (b"pdf-data", "application/pdf")

        images = [_make_image(), _make_image()]
        mock_b2i.return_value = images

        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.transcribe_image.side_effect = [
            ("Page 1 text", "model-a"),
            ("Page 2 text", "model-a"),
        ]

        mock_assemble.return_value = ("Full text\n\nTranscribed by model: model-a", {"model-a"})

        proc = DocumentProcessor(
            {"id": 1, "title": "Test", "tags": [443]},
            paperless,
            ocr_provider,
            settings,
        )

        # Act
        proc.process()

        # Assert — full pipeline invoked with correct data flow
        mock_claim.assert_called_once()
        paperless.download_content.assert_called_once_with(1)
        mock_b2i.assert_called_once()
        mock_assemble.assert_called_once()
        mock_release.assert_called_once()
        # Verify update_document called with correct content and tags
        paperless.update_document.assert_called_once()
        call_args = paperless.update_document.call_args[0]
        assert call_args[0] == 1  # doc_id
        assert "Full text" in call_args[1]  # OCR text
        tags = call_args[2]
        assert 444 in tags  # POST_TAG_ID added
        assert 443 not in tags  # PRE_TAG_ID removed


# -----------------------------------------------------------------------
# process() — claim fails
# -----------------------------------------------------------------------

class TestProcessClaimFails:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=False)
    def test_claim_fails_returns_early(self, mock_claim, mock_release):
        # Arrange
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}
        proc = _make_processor(paperless=paperless)

        # Act
        proc.process()

        # Assert — no download, no OCR
        paperless.download_content.assert_not_called()
        # Release not called because claimed=False
        mock_release.assert_not_called()


# -----------------------------------------------------------------------
# process() — error tag present
# -----------------------------------------------------------------------

class TestProcessErrorTagPresent:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag")
    @patch("ocr.worker.clean_pipeline_tags")
    def test_error_tag_skips_ocr(self, mock_clean, mock_claim, mock_release):
        # Arrange
        settings = _make_settings(ERROR_TAG_ID=552)
        paperless = _make_mock_paperless()
        # Return doc with error tag
        paperless.get_document.return_value = {
            "id": 1, "title": "T", "tags": [443, 552]
        }
        mock_clean.return_value = {552}

        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc.process()

        # Assert — claim never called, finalize_with_error called via clean_pipeline_tags
        mock_claim.assert_not_called()
        paperless.download_content.assert_not_called()
        mock_clean.assert_called_once()
        paperless.update_document_metadata.assert_called_once()


# -----------------------------------------------------------------------
# process() — refresh failure
# -----------------------------------------------------------------------

class TestProcessRefreshFailure:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag")
    def test_refresh_failure_returns(self, mock_claim, mock_release):
        # Arrange
        paperless = _make_mock_paperless()
        paperless.get_document.side_effect = Exception("Network error")
        proc = _make_processor(paperless=paperless)

        # Act
        proc.process()

        # Assert
        mock_claim.assert_not_called()
        paperless.download_content.assert_not_called()
        mock_release.assert_not_called()


# -----------------------------------------------------------------------
# process() — image conversion failure
# -----------------------------------------------------------------------

class TestProcessImageConversionFailure:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images", side_effect=RuntimeError("Bad image"))
    @patch("ocr.worker.clean_pipeline_tags")
    def test_conversion_failure_finalizes_error(
        self, mock_clean, mock_b2i, mock_claim, mock_release
    ):
        # Arrange
        settings = _make_settings(
            OCR_PROCESSING_TAG_ID=999,
            ERROR_TAG_ID=552,
        )
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}
        mock_clean.return_value = {552}

        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc.process()

        # Assert — lock released
        mock_release.assert_called_once()


# -----------------------------------------------------------------------
# process() — always releases lock in finally
# -----------------------------------------------------------------------

class TestProcessAlwaysReleasesLock:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images")
    def test_lock_released_on_download_failure(
        self, mock_b2i, mock_claim, mock_release
    ):
        # Arrange
        settings = _make_settings(OCR_PROCESSING_TAG_ID=999)
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}
        paperless.download_content.side_effect = Exception("Download failed")

        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        with pytest.raises(Exception, match="Download failed"):
            proc.process()

        # Assert — lock still released
        mock_release.assert_called_once()

    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images")
    def test_lock_released_on_ocr_failure(
        self, mock_b2i, mock_claim, mock_release
    ):
        # Arrange
        settings = _make_settings(OCR_PROCESSING_TAG_ID=999)
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}
        images = [_make_image()]
        mock_b2i.return_value = images

        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.transcribe_image.side_effect = Exception("OCR boom")

        proc = _make_processor(
            paperless=paperless, ocr_provider=ocr_provider, settings=settings
        )

        # Act — the exception from the thread pool may propagate
        # process() catches via ThreadPoolExecutor, page results get error marker
        proc.process()

        # Assert
        mock_release.assert_called_once()


# -----------------------------------------------------------------------
# _bytes_to_images
# -----------------------------------------------------------------------

class TestBytesToImages:
    @patch("ocr.worker.bytes_to_images")
    def test_delegates_to_image_converter(self, mock_b2i):
        # Arrange
        mock_b2i.return_value = [_make_image()]
        settings = _make_settings(OCR_DPI=200)
        proc = _make_processor(settings=settings)

        # Act
        result = proc._bytes_to_images(b"data", "image/png")

        # Assert
        mock_b2i.assert_called_once_with(b"data", "image/png", dpi=200)
        assert len(result) == 1


# -----------------------------------------------------------------------
# _ocr_pages_in_parallel
# -----------------------------------------------------------------------

class TestOcrPagesInParallel:
    def test_preserves_page_order(self):
        # Arrange — single worker for deterministic side_effect ordering
        settings = _make_settings(PAGE_WORKERS=1)
        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.transcribe_image.side_effect = [
            ("Text page 1", "m"),
            ("Text page 2", "m"),
            ("Text page 3", "m"),
        ]
        proc = _make_processor(ocr_provider=ocr_provider, settings=settings)
        images = [_make_image() for _ in range(3)]

        # Act
        results, failed = proc._ocr_pages_in_parallel(images)

        # Assert — correct count, no failures, and content in correct order
        assert len(results) == 3
        assert failed == []
        assert results[0] == ("Text page 1", "m")
        assert results[1] == ("Text page 2", "m")
        assert results[2] == ("Text page 3", "m")

    def test_handles_ocr_exception_per_page(self):
        # Arrange
        settings = _make_settings(PAGE_WORKERS=1)
        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.transcribe_image.side_effect = [
            ("Text page 1", "m"),
            Exception("OCR failed on page 2"),
            ("Text page 3", "m"),
        ]
        proc = _make_processor(ocr_provider=ocr_provider, settings=settings)
        images = [_make_image() for _ in range(3)]

        # Act
        results, failed = proc._ocr_pages_in_parallel(images)

        # Assert
        assert len(results) == 3
        assert 2 in failed  # page 2 (1-indexed)
        assert OCR_ERROR_MARKER in results[1][0]

    def test_empty_images_list(self):
        # Arrange
        proc = _make_processor()

        # Act
        results, failed = proc._ocr_pages_in_parallel([])

        # Assert
        assert results == []
        assert failed == []


# -----------------------------------------------------------------------
# _assemble_full_text
# -----------------------------------------------------------------------

class TestAssembleFullText:
    @patch("ocr.worker.assemble_full_text")
    def test_delegates_to_text_assembly(self, mock_assemble):
        # Arrange
        mock_assemble.return_value = ("assembled", {"m1"})
        settings = _make_settings(OCR_INCLUDE_PAGE_MODELS=True)
        proc = _make_processor(settings=settings)
        page_results = [("text", "m1")]

        # Act
        text, models = proc._assemble_full_text(1, page_results)

        # Assert
        mock_assemble.assert_called_once_with(
            1, page_results, include_page_models=True
        )
        assert text == "assembled"
        assert models == {"m1"}

    @patch("ocr.worker.assemble_full_text")
    def test_include_page_models_false(self, mock_assemble):
        # Arrange
        mock_assemble.return_value = ("text", set())
        settings = _make_settings(OCR_INCLUDE_PAGE_MODELS=False)
        proc = _make_processor(settings=settings)

        # Act
        proc._assemble_full_text(1, [])

        # Assert
        mock_assemble.assert_called_once_with(
            1, [], include_page_models=False
        )


# -----------------------------------------------------------------------
# _update_paperless_document — happy path
# -----------------------------------------------------------------------

class TestUpdatePaperlessDocumentHappy:
    @patch("ocr.worker.get_latest_tags")
    def test_happy_path_swaps_tags(self, mock_get_tags):
        # Arrange
        settings = _make_settings(
            PRE_TAG_ID=443,
            POST_TAG_ID=444,
            OCR_PROCESSING_TAG_ID=999,
        )
        paperless = _make_mock_paperless()
        mock_get_tags.return_value = {443, 999, 100}  # user tag 100

        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._update_paperless_document("Good OCR text", {"model-a"})

        # Assert
        paperless.update_document.assert_called_once()
        args = paperless.update_document.call_args
        doc_id, text, tags = args[0]
        assert doc_id == 1
        assert text == "Good OCR text"
        tag_set = set(tags)
        assert 443 not in tag_set  # pre removed
        assert 999 not in tag_set  # processing removed
        assert 444 in tag_set      # post added
        assert 100 in tag_set      # user tag preserved


# -----------------------------------------------------------------------
# _update_paperless_document — error conditions
# -----------------------------------------------------------------------

class TestUpdatePaperlessDocumentErrors:
    @patch("ocr.worker.get_latest_tags", return_value={443})
    @patch("ocr.worker.clean_pipeline_tags", return_value=set())
    def test_empty_text_marks_error(self, mock_clean, mock_get_tags):
        # Arrange
        settings = _make_settings(ERROR_TAG_ID=552)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._update_paperless_document("   ", set())

        # Assert — finalize_with_error calls update_document with error tag
        paperless.update_document.assert_called_once()
        tags_arg = paperless.update_document.call_args[0][2]
        assert 552 in tags_arg

    @patch("ocr.worker.get_latest_tags", return_value={443})
    @patch("ocr.worker.clean_pipeline_tags", return_value=set())
    def test_ocr_error_marker_in_text_marks_error(self, mock_clean, mock_get_tags):
        # Arrange
        settings = _make_settings(ERROR_TAG_ID=552)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._update_paperless_document(
            f"Some text {OCR_ERROR_MARKER} more", {"m"}
        )

        # Assert — finalize_with_error calls update_document with error tag
        paperless.update_document.assert_called_once()
        tags_arg = paperless.update_document.call_args[0][2]
        assert 552 in tags_arg

    @patch("ocr.worker.get_latest_tags", return_value={443})
    @patch("ocr.worker.clean_pipeline_tags", return_value=set())
    def test_refusal_mark_in_text_marks_error(self, mock_clean, mock_get_tags):
        # Arrange
        settings = _make_settings(
            ERROR_TAG_ID=552,
            REFUSAL_MARK="CHATGPT REFUSED TO TRANSCRIBE",
        )
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._update_paperless_document(
            "CHATGPT REFUSED TO TRANSCRIBE", set()
        )

        # Assert — finalize_with_error calls update_document with error tag
        paperless.update_document.assert_called_once()
        tags_arg = paperless.update_document.call_args[0][2]
        assert 552 in tags_arg

    @patch("ocr.worker.get_latest_tags", return_value={443})
    @patch("ocr.worker.clean_pipeline_tags", return_value=set())
    def test_redacted_marker_in_text_marks_error(self, mock_clean, mock_get_tags):
        # Arrange
        settings = _make_settings(ERROR_TAG_ID=552)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._update_paperless_document("Name: [REDACTED]", {"m"})

        # Assert — finalize_with_error calls update_document with error tag
        paperless.update_document.assert_called_once()
        tags_arg = paperless.update_document.call_args[0][2]
        assert 552 in tags_arg


# -----------------------------------------------------------------------
# _finalize_with_error
# -----------------------------------------------------------------------

class TestFinalizeWithError:
    @patch("ocr.worker.clean_pipeline_tags")
    def test_adds_error_tag(self, mock_clean):
        # Arrange
        mock_clean.return_value = {100}  # user tag preserved
        settings = _make_settings(ERROR_TAG_ID=552)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._finalize_with_error({443, 100})

        # Assert
        paperless.update_document_metadata.assert_called_once()
        call_tags = set(paperless.update_document_metadata.call_args[1]["tags"])
        assert 552 in call_tags  # error tag added
        assert 100 in call_tags  # user tag preserved

    @patch("ocr.worker.clean_pipeline_tags")
    def test_no_error_tag_configured(self, mock_clean):
        # Arrange
        mock_clean.return_value = {100}
        settings = _make_settings(ERROR_TAG_ID=None)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._finalize_with_error({443, 100})

        # Assert — still cleans tags, no error tag added
        paperless.update_document_metadata.assert_called_once()
        call_tags = set(paperless.update_document_metadata.call_args[1]["tags"])
        assert 100 in call_tags

    @patch("ocr.worker.clean_pipeline_tags")
    def test_with_content_updates_document(self, mock_clean):
        # Arrange
        mock_clean.return_value = {552}
        settings = _make_settings(ERROR_TAG_ID=552)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._finalize_with_error({443}, content="Error OCR text")

        # Assert — uses update_document (not update_document_metadata)
        paperless.update_document.assert_called_once()
        args = paperless.update_document.call_args[0]
        assert args[1] == "Error OCR text"

    @patch("ocr.worker.clean_pipeline_tags")
    def test_without_content_uses_metadata_update(self, mock_clean):
        # Arrange
        mock_clean.return_value = set()
        settings = _make_settings(ERROR_TAG_ID=None)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        proc._finalize_with_error({443})

        # Assert
        paperless.update_document_metadata.assert_called_once()
        paperless.update_document.assert_not_called()


# -----------------------------------------------------------------------
# _log_ocr_stats
# -----------------------------------------------------------------------

class TestLogOcrStats:
    @patch("ocr.worker.log")
    def test_normal_stats_logged(self, mock_log):
        # Arrange
        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.get_stats.return_value = {
            "attempts": 5,
            "refusals": 1,
            "api_errors": 0,
            "fallback_successes": 1,
        }
        proc = _make_processor(ocr_provider=ocr_provider)

        # Act
        proc._log_ocr_stats()

        # Assert — stats fetched and logged
        ocr_provider.get_stats.assert_called_once()
        mock_log.info.assert_called_once()
        log_kwargs = mock_log.info.call_args.kwargs
        assert log_kwargs["attempts"] == 5

    @patch("ocr.worker.log")
    def test_zero_attempts_not_logged(self, mock_log):
        # Arrange — get_stats returns 0 attempts
        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.get_stats.return_value = {
            "attempts": 0,
            "refusals": 0,
            "api_errors": 0,
            "fallback_successes": 0,
        }
        proc = _make_processor(ocr_provider=ocr_provider)

        # Act — should return early without logging
        proc._log_ocr_stats()

        # Assert — stats fetched but NOT logged (zero attempts)
        ocr_provider.get_stats.assert_called_once()
        mock_log.info.assert_not_called()

    def test_no_get_stats_method(self):
        # Arrange — provider without get_stats
        ocr_provider = MagicMock(spec=[])  # no attributes
        proc = _make_processor(ocr_provider=ocr_provider)

        # Act — should not raise
        proc._log_ocr_stats()

    def test_empty_stats_dict(self):
        # Arrange
        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.get_stats.return_value = {}
        proc = _make_processor(ocr_provider=ocr_provider)

        # Act — should not raise (empty dict is falsy -> returns early)
        proc._log_ocr_stats()


# -----------------------------------------------------------------------
# Images always closed
# -----------------------------------------------------------------------

class TestImagesAlwaysClosed:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images")
    @patch("ocr.worker.assemble_full_text", return_value=("text", {"m"}))
    @patch("ocr.worker.get_latest_tags", return_value={443})
    def test_images_closed_on_success(
        self, mock_tags, mock_assemble, mock_b2i, mock_claim, mock_release
    ):
        # Arrange
        img1 = MagicMock(spec=Image.Image)
        img2 = MagicMock(spec=Image.Image)
        mock_b2i.return_value = [img1, img2]

        settings = _make_settings(OCR_PROCESSING_TAG_ID=None)
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}
        ocr_provider = _make_mock_ocr_provider()

        proc = _make_processor(
            paperless=paperless, ocr_provider=ocr_provider, settings=settings
        )

        # Act
        proc.process()

        # Assert
        img1.close.assert_called_once()
        img2.close.assert_called_once()

    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images")
    def test_images_closed_on_ocr_error(
        self, mock_b2i, mock_claim, mock_release
    ):
        # Arrange
        img1 = MagicMock(spec=Image.Image)
        mock_b2i.return_value = [img1]

        settings = _make_settings(OCR_PROCESSING_TAG_ID=999)
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}

        ocr_provider = _make_mock_ocr_provider()
        ocr_provider.transcribe_image.side_effect = Exception("OCR boom")

        proc = _make_processor(
            paperless=paperless, ocr_provider=ocr_provider, settings=settings
        )

        # Act
        proc.process()

        # Assert — image closed despite OCR error
        img1.close.assert_called_once()


# -----------------------------------------------------------------------
# _refresh_document
# -----------------------------------------------------------------------

class TestRefreshDocument:
    def test_success_returns_latest(self):
        # Arrange
        paperless = _make_mock_paperless()
        latest_doc = {"id": 1, "title": "Updated", "tags": [443]}
        paperless.get_document.return_value = latest_doc
        proc = _make_processor(paperless=paperless)

        # Act
        result = proc._refresh_document()

        # Assert
        assert result == latest_doc
        assert proc.doc == latest_doc

    def test_failure_returns_none(self):
        # Arrange
        paperless = _make_mock_paperless()
        paperless.get_document.side_effect = Exception("Network error")
        proc = _make_processor(paperless=paperless)

        # Act
        result = proc._refresh_document()

        # Assert
        assert result is None


# -----------------------------------------------------------------------
# _claim_processing_tag
# -----------------------------------------------------------------------

class TestClaimProcessingTag:
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    def test_delegates_to_claim_function(self, mock_claim):
        # Arrange
        settings = _make_settings(OCR_PROCESSING_TAG_ID=999)
        paperless = _make_mock_paperless()
        proc = _make_processor(paperless=paperless, settings=settings)

        # Act
        result = proc._claim_processing_tag()

        # Assert
        assert result is True
        mock_claim.assert_called_once_with(
            paperless_client=paperless,
            doc_id=1,
            tag_id=999,
            purpose="ocr",
        )

    @patch("ocr.worker.claim_processing_tag", return_value=False)
    def test_returns_false_on_failure(self, mock_claim):
        # Arrange
        proc = _make_processor()

        # Act
        result = proc._claim_processing_tag()

        # Assert
        assert result is False


# -----------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------

class TestDocumentProcessorInit:
    def test_extracts_doc_id(self):
        # Arrange
        from tests.helpers.factories import make_document
        doc = make_document(id=42)

        # Act
        proc = _make_processor(doc=doc)

        # Assert
        assert proc.doc_id == 42

    def test_title_defaults_to_untitled(self):
        # Arrange
        from tests.helpers.factories import make_document
        doc = make_document(title=None)

        # Act
        proc = _make_processor(doc=doc)

        # Assert
        assert proc.title == "<untitled>"

    def test_title_from_doc(self):
        # Arrange
        from tests.helpers.factories import make_document
        doc = make_document(title="My Document")

        # Act
        proc = _make_processor(doc=doc)

        # Assert
        assert proc.title == "My Document"


# -----------------------------------------------------------------------
# process() — no pages
# -----------------------------------------------------------------------

class TestProcessNoPages:
    @patch("ocr.worker.release_processing_tag")
    @patch("ocr.worker.claim_processing_tag", return_value=True)
    @patch("ocr.worker.bytes_to_images", return_value=[])
    def test_no_pages_returns_early(self, mock_b2i, mock_claim, mock_release):
        # Arrange
        settings = _make_settings(OCR_PROCESSING_TAG_ID=999)
        paperless = _make_mock_paperless()
        paperless.get_document.return_value = {"id": 1, "title": "T", "tags": [443]}
        ocr_provider = _make_mock_ocr_provider()

        proc = _make_processor(
            paperless=paperless, ocr_provider=ocr_provider, settings=settings
        )

        # Act
        proc.process()

        # Assert — no OCR attempted
        ocr_provider.transcribe_image.assert_not_called()
        # Lock still released
        mock_release.assert_called_once()
