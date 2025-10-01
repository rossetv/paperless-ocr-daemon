import os
from unittest.mock import MagicMock, call

import pytest
from PIL import Image

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
    # Simulate downloading a file to a temporary path
    client.download_file.return_value = ("/tmp/test_doc.pdf", "application/pdf")
    return client


@pytest.fixture
def mock_ocr_provider(mocker):
    """Fixture to create a mock OcrProvider."""
    provider = MagicMock()
    # Simulate OCR results for two pages
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


@pytest.mark.parametrize(
    "file_to_images_mock",
    [[create_test_image(), create_test_image()]],
    indirect=True,
)
def test_process_document_success(
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
    file_to_images_mock,
):
    """
    Test the successful end-to-end processing of a document.
    """
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    processor.process()

    # 1. Verify download was called
    mock_paperless_client.download_file.assert_called_once_with(1)

    # 2. Verify OCR was called for each page
    assert mock_ocr_provider.transcribe_image.call_count == 2

    # 3. Verify document update was called with correct content and tags
    expected_content = (
        "--- Page 1 ---\n"
        "Page 1 text\n\n"
        "--- Page 2 ---\n"
        "Page 2 text\n\n"
        "Transcribed by model: model-a, model-b"
    )
    # The worker calculates the sorted list of models before creating the string
    expected_tags = [11]  # Post-OCR tag, pre-OCR tag removed
    mock_paperless_client.update_document.assert_called_once()
    args, _ = mock_paperless_client.update_document.call_args
    assert args[0] == 1
    # Sort models in expected content to match implementation
    assert "Transcribed by model: model-a, model-b" in args[1]
    assert sorted(args[2]) == sorted(expected_tags)


@pytest.mark.parametrize("file_to_images_mock", [[create_test_image()]], indirect=True)
def test_process_single_page_document(
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
    file_to_images_mock,
):
    """
    Test that page headers are omitted for single-page documents.
    """
    # Adjust mock for a single page
    mock_ocr_provider.transcribe_image.side_effect = [("Single page text", "model-a")]

    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    processor.process()

    expected_content = "Single page text\n\nTranscribed by model: model-a"
    mock_paperless_client.update_document.assert_called_once_with(
        1, expected_content, [11]
    )


@pytest.mark.parametrize("file_to_images_mock", [[]], indirect=True)
def test_process_document_with_no_images(
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
    file_to_images_mock,
):
    """
    Test that the process handles documents that produce no images.
    """
    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )
    processor.process()

    # OCR and update should not be called
    mock_ocr_provider.transcribe_image.assert_not_called()
    mock_paperless_client.update_document.assert_not_called()
    # Cleanup should still be called
    file_to_images_mock.cleanup.assert_called_once()


@pytest.mark.parametrize("file_to_images_mock", [[create_test_image()]], indirect=True)
def test_cleanup_on_error(
    settings,
    mock_paperless_client,
    mock_ocr_provider,
    mock_doc,
    file_to_images_mock,
    caplog,
):
    """
    Test that the temporary file is cleaned up and the error is logged
    if an exception occurs during page processing.
    """
    # Simulate an error during OCR
    mock_ocr_provider.transcribe_image.side_effect = ValueError("OCR failed!")

    processor = DocumentProcessor(
        mock_doc, mock_paperless_client, mock_ocr_provider, settings
    )

    # The process should not raise an exception, but log it instead
    processor.process()

    # 1. Assert that the error was logged
    assert "OCR failed on page 1: OCR failed!" in caplog.text

    # 2. Assert that the process continued and updated the document (with no text)
    mock_paperless_client.update_document.assert_called_once_with(1, "", [11])

    # 3. Assert that cleanup was still called
    file_to_images_mock.cleanup.assert_called_once()


# --- Fixture to mock file conversion and cleanup ---


@pytest.fixture
def file_to_images_mock(mocker, request):
    """
    Mocks the _file_to_images and _cleanup_temp_file methods of the DocumentProcessor.
    This allows testing the orchestration logic without touching the file system
    or external libraries like pdf2image.
    """
    images = request.param
    mock_file_to_images = mocker.patch(
        "paperless_ocr.worker.DocumentProcessor._file_to_images",
        return_value=images,
    )
    mock_cleanup = mocker.patch(
        "paperless_ocr.worker.DocumentProcessor._cleanup_temp_file"
    )

    # Attach the cleanup mock to the file_to_images mock so it can be accessed in tests
    mock_file_to_images.cleanup = mock_cleanup
    return mock_file_to_images