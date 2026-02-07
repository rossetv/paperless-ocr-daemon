import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, UnidentifiedImageError

from paperless_ocr.config import Settings
from paperless_ocr.worker import DocumentProcessor


@pytest.fixture
def settings(mocker):
    """Fixture to create a Settings object for tests."""
    mocker.patch.dict(
        os.environ,
        {
            "PAPERLESS_TOKEN": "test_token",
            "OPENAI_API_KEY": "test_api_key",
            "PRE_TAG_ID": "10",
            "POST_TAG_ID": "11",
        },
        clear=True,
    )
    return Settings()


@pytest.fixture
def mock_paperless_client(mocker):
    """Fixture to create a mock PaperlessClient."""
    client = MagicMock()
    client.download_content.return_value = (b"file_content", "application/pdf")
    return client


@pytest.fixture
def mock_ocr_provider(mocker):
    """Fixture to create a mock OcrProvider."""
    provider = MagicMock()
    provider.transcribe_image.side_effect = [
        ("Page 1 text", "model-a"),
        ("Page 2 text", "model-b"),
    ]
    return provider


@pytest.fixture
def mock_doc():
    """Fixture to create a sample document dictionary."""
    return {"id": 1, "title": "Test Document", "tags": [10]}


def create_test_image():
    """Helper to create a test image."""
    return Image.new("RGB", (100, 100), "black")


@patch("paperless_ocr.worker.convert_from_path", return_value=[create_test_image(), create_test_image()])
@patch("tempfile.NamedTemporaryFile")
def test_process_document_success(
    mock_tempfile,
    mock_convert,
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
):
    """
    Test the successful end-to-end processing of a document, ensuring
    proper file handling and orchestration.
    """
    # Mock the temporary file context manager
    mock_file = MagicMock()
    mock_file.__enter__.return_value.name = "/tmp/test.pdf"
    mock_tempfile.return_value = mock_file

    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    processor.process()

    # 1. Verify content was downloaded
    mock_paperless_client.download_content.assert_called_once_with(1)

    # 2. Verify temp file was written to
    mock_file.__enter__.return_value.write.assert_called_once_with(b"file_content")
    mock_file.__enter__.return_value.flush.assert_called_once()

    # 3. Verify file was converted to images
    mock_convert.assert_called_once_with("/tmp/test.pdf", dpi=settings.OCR_DPI)

    # 4. Verify OCR was called for each page
    assert mock_ocr_provider.transcribe_image.call_count == 2

    # 5. Verify document was updated with correct content and tags
    expected_content = (
        "--- Page 1 ---\n"
        "Page 1 text\n\n"
        "--- Page 2 ---\n"
        "Page 2 text\n\n"
        "Transcribed by model: model-a, model-b"
    )
    mock_paperless_client.update_document.assert_called_once()
    args, _ = mock_paperless_client.update_document.call_args
    assert args[0] == 1
    assert "Transcribed by model: model-a, model-b" in args[1]
    assert sorted(args[2]) == [11]


@patch("paperless_ocr.worker.convert_from_path", return_value=[])
@patch("tempfile.NamedTemporaryFile")
def test_process_document_with_no_images(
    mock_tempfile,
    mock_convert,
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
):
    """
    Test that the process handles documents that produce no images and still
    cleans up correctly.
    """
    mock_file = MagicMock()
    mock_tempfile.return_value = mock_file

    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    processor.process()

    # Verify download and file write still happened
    mock_paperless_client.download_content.assert_called_once_with(1)
    mock_file.__enter__.return_value.write.assert_called_once_with(b"file_content")

    # OCR and update should not be called
    mock_ocr_provider.transcribe_image.assert_not_called()
    mock_paperless_client.update_document.assert_not_called()

    # The with statement for the temp file should have completed, ensuring cleanup
    mock_file.__exit__.assert_called_once()


@patch("paperless_ocr.worker.convert_from_path")
@patch("tempfile.NamedTemporaryFile")
def test_resource_cleanup_on_processing_error(
    mock_tempfile,
    mock_convert,
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
):
    """
    Test that the temporary file is cleaned up via the 'with' statement
    even if an error occurs during image conversion.
    """
    mock_file = MagicMock()
    mock_tempfile.return_value = mock_file
    mock_convert.side_effect = Exception("PDF conversion failed!")

    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )

    with pytest.raises(Exception, match="PDF conversion failed!"):
        processor.process()

    # Assert that the temporary file's context manager was exited,
    # which guarantees cleanup by tempfile.NamedTemporaryFile.
    mock_file.__exit__.assert_called_once()


def test_file_to_images_non_pdf(settings, mock_paperless_client, mock_ocr_provider, mock_doc, mocker):
    """
    Test that non-PDF content is opened as a single image.
    """
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    image = create_test_image()
    open_mock = mocker.patch("paperless_ocr.worker.Image.open", return_value=image)

    images = processor._file_to_images("/tmp/test.png", "image/png")

    open_mock.assert_called_once_with("/tmp/test.png")
    assert images == [image]


def test_file_to_images_invalid_image(settings, mock_paperless_client, mock_ocr_provider, mock_doc, mocker):
    """
    Test that invalid images raise a RuntimeError.
    """
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    mocker.patch(
        "paperless_ocr.worker.Image.open", side_effect=UnidentifiedImageError("bad")
    )

    with pytest.raises(RuntimeError, match="Unable to open image"):
        processor._file_to_images("/tmp/bad.bin", "image/png")


def test_ocr_pages_in_parallel_handles_errors_and_order(
    settings, mock_paperless_client, mock_doc, mocker
):
    """
    Test that OCR results preserve order and errors are handled per page.
    """
    settings.PAGE_WORKERS = 2
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, MagicMock(), settings
    )
    images = [create_test_image(), create_test_image(), create_test_image()]
    for index, image in enumerate(images):
        image.info["index"] = index

    def side_effect(image, *args, **kwargs):
        index = image.info["index"]
        if index == 1:
            raise RuntimeError("ocr failed")
        return (f"text-{index}", f"model-{index}")

    processor.ocr_provider.transcribe_image.side_effect = side_effect

    results, failed_pages = processor._ocr_pages_in_parallel(images)

    assert results[0] == ("text-0", "model-0")
    assert results[1] == ("", "")
    assert results[2] == ("text-2", "model-2")
    assert failed_pages == [2]


@patch("paperless_ocr.worker.convert_from_path", return_value=[create_test_image(), create_test_image()])
@patch("tempfile.NamedTemporaryFile")
def test_process_document_partial_failure_marks_error_tag(
    mock_tempfile,
    mock_convert,
    settings,
    mock_paperless_client,
    mock_doc,
):
    """
    Test that partial OCR failures add the error tag and skip document updates.
    """
    mock_paperless_client.get_document.return_value = mock_doc
    mock_paperless_client.download_content.return_value = (b"file_content", "application/pdf")
    mock_paperless_client.update_document_metadata.reset_mock()

    mock_file = MagicMock()
    mock_file.__enter__.return_value.name = "/tmp/test.pdf"
    mock_tempfile.return_value = mock_file

    ocr_provider = MagicMock()
    ocr_provider.transcribe_image.side_effect = [
        ("Page 1 text", "model-a"),
        RuntimeError("ocr failed"),
    ]

    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, ocr_provider, settings
    )
    processor.process()

    mock_paperless_client.update_document.assert_not_called()
    mock_paperless_client.update_document_metadata.assert_called()
    args, kwargs = mock_paperless_client.update_document_metadata.call_args
    assert args[0] == 1
    assert settings.ERROR_TAG_ID in kwargs["tags"]


def test_assemble_full_text_skips_blank_pages(settings, mock_paperless_client, mock_ocr_provider, mock_doc):
    """
    Test that blank pages are skipped and headers/footers are applied correctly.
    """
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    images = [create_test_image(), create_test_image(), create_test_image()]
    page_results = [
        ("Page 1 text", "model-a"),
        ("   ", ""),
        ("Page 3 text", "model-b"),
    ]

    full_text, models_used = processor._assemble_full_text(images, page_results)

    expected = (
        "--- Page 1 ---\n"
        "Page 1 text\n\n"
        "--- Page 3 ---\n"
        "Page 3 text\n\n"
        "Transcribed by model: model-a, model-b"
    )
    assert full_text == expected
    assert models_used == {"model-a", "model-b"}


def test_assemble_full_text_includes_page_models(
    settings, mock_paperless_client, mock_ocr_provider, mock_doc
):
    """
    Test that page headers include model names when enabled.
    """
    settings.OCR_INCLUDE_PAGE_MODELS = True
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    images = [create_test_image(), create_test_image()]
    page_results = [
        ("Page 1 text", "model-a"),
        ("Page 2 text", ""),
    ]

    full_text, models_used = processor._assemble_full_text(images, page_results)

    expected = (
        "--- Page 1 (model-a) ---\n"
        "Page 1 text\n\n"
        "--- Page 2 ---\n"
        "Page 2 text\n\n"
        "Transcribed by model: model-a"
    )
    assert full_text == expected
    assert models_used == {"model-a"}


def test_update_paperless_document_tags(settings, mock_paperless_client, mock_ocr_provider):
    """
    Test that pre-tag is removed, post-tag added, and other tags preserved.
    """
    doc = {"id": 5, "title": "Doc", "tags": [settings.PRE_TAG_ID, 42, 42]}
    processor = DocumentProcessor(doc, mock_paperless_client, mock_ocr_provider, settings)

    processor._update_paperless_document("text", {"model-a"})

    args, _ = mock_paperless_client.update_document.call_args
    assert args[0] == 5
    assert args[1] == "text"
    assert set(args[2]) == {settings.POST_TAG_ID, 42}
