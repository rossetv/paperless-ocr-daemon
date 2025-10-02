import os
from unittest.mock import MagicMock, call, patch

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