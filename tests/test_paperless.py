import os

import pytest
import requests
import requests_mock

from paperless_ocr.config import Settings
from paperless_ocr.paperless import PaperlessClient


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
            "MAX_RETRIES": "3",
        },
        clear=True,
    )
    return Settings()


@pytest.fixture
def paperless_client(settings):
    """Fixture to create a PaperlessClient instance."""
    return PaperlessClient(settings)


def test_get_documents_to_process(paperless_client, settings, requests_mock):
    """
    Test that the client correctly fetches all documents with the pre-OCR tag.
    """
    # Mock the paginated API response
    # Page 1
    requests_mock.get(
        f"{settings.PAPERLESS_URL}/api/documents/?tags__id=10&page_size=100",
        json={
            "next": f"{settings.PAPERLESS_URL}/api/documents/page2",
            "results": [
                {"id": 1, "tags": [10]},
                {"id": 2, "tags": [10, 11]},
            ],
        },
    )
    # Page 2
    requests_mock.get(
        f"{settings.PAPERLESS_URL}/api/documents/page2",
        json={
            "next": None,
            "results": [
                {"id": 3, "tags": [10]},
            ],
        },
    )

    docs = list(paperless_client.get_documents_to_process())

    assert len(docs) == 3
    assert [d["id"] for d in docs] == [1, 2, 3]


def test_download_content(paperless_client, settings, requests_mock):
    """
    Test that file content is correctly downloaded.
    """
    doc_id = 1
    file_content = b"This is a test PDF."
    requests_mock.get(
        f"{settings.PAPERLESS_URL}/api/documents/{doc_id}/download/",
        content=file_content,
        headers={"Content-Type": "application/pdf"},
    )

    content, content_type = paperless_client.download_content(doc_id)

    assert content_type == "application/pdf"
    assert content == file_content


def test_download_content_defaults_content_type(paperless_client, settings, requests_mock):
    """
    Test that the content type defaults when not provided by the server.
    """
    doc_id = 2
    file_content = b"binary data"
    requests_mock.get(
        f"{settings.PAPERLESS_URL}/api/documents/{doc_id}/download/",
        content=file_content,
        headers={},
    )

    content, content_type = paperless_client.download_content(doc_id)

    assert content_type == "application/pdf"
    assert content == file_content


def test_update_document(paperless_client, settings, requests_mock):
    """
    Test that a document is correctly updated with new content and tags.
    """
    doc_id = 1
    new_content = "This is the OCR text."
    new_tags = [11, 12]

    # Mock the PATCH request and check the payload
    mock_patch = requests_mock.patch(
        f"{settings.PAPERLESS_URL}/api/documents/{doc_id}/", status_code=200
    )

    paperless_client.update_document(doc_id, new_content, new_tags)

    assert mock_patch.called
    assert mock_patch.last_request.json() == {
        "content": new_content,
        "tags": new_tags,
    }


def test_retry_on_network_error(paperless_client, settings, requests_mock):
    """
    Test that the client retries on a transient network error.
    """
    url = f"{settings.PAPERLESS_URL}/api/documents/?tags__id=10&page_size=100"
    requests_mock.get(
        url,
        [
            {"exc": requests.exceptions.ConnectionError},
            {"exc": requests.exceptions.Timeout},
            {"json": {"next": None, "results": [{"id": 1, "tags": [10]}]}},
        ],
    )

    # The call should succeed on the third attempt
    docs = list(paperless_client.get_documents_to_process())
    assert len(docs) == 1
    assert requests_mock.call_count == 3


def test_retry_fails_after_max_retries(paperless_client, settings, requests_mock):
    """
    Test that the client gives up after exhausting all retries.
    """
    url = f"{settings.PAPERLESS_URL}/api/documents/?tags__id=10&page_size=100"
    requests_mock.get(url, exc=requests.exceptions.ConnectionError)

    with pytest.raises(requests.exceptions.ConnectionError):
        list(paperless_client.get_documents_to_process())

    assert requests_mock.call_count == settings.MAX_RETRIES
