"""
Paperless-ngx API Client
========================

This module provides a client for interacting with the Paperless-ngx API.
It encapsulates all the logic for making authenticated requests to a
Paperless-ngx instance, handling pagination, and performing common
operations like listing documents, downloading files, and updating
document content and tags.

The `PaperlessClient` class is designed to be a reusable and testable
component that abstracts away the details of the Paperless-ngx REST API.
"""

from typing import Generator, Iterable

import requests

from .config import Settings
from .utils import retry


class PaperlessClient:
    """A client for interacting with the Paperless-ngx API."""

    def __init__(self, settings: Settings):
        """Initializes the client with a session and authentication."""
        self.settings = settings
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Token {self.settings.PAPERLESS_TOKEN}"}
        )

    @retry(retryable_exceptions=(requests.exceptions.RequestException,))
    def _get(self, *args, **kwargs) -> requests.Response:
        """A retriable version of session.get."""
        return self._session.get(*args, **kwargs)

    @retry(retryable_exceptions=(requests.exceptions.RequestException,))
    def _patch(self, *args, **kwargs) -> requests.Response:
        """A retriable version of session.patch."""
        return self._session.patch(*args, **kwargs)

    def _list_all(self, url: str) -> Generator[dict, None, None]:
        """
        Generator that follows Paperless-ngx paginated API and yields every result.
        """
        while url:
            response = self._get(url)
            response.raise_for_status()
            page = response.json()
            yield from page.get("results", [])
            url = page.get("next")

    def get_documents_to_process(self) -> Iterable[dict]:
        """
        Return documents that have the pre-OCR tag and do not yet have the post-OCR tag.
        """
        url = f"{self.settings.PAPERLESS_URL}/api/documents/?tags__id={self.settings.PRE_TAG_ID}&page_size=100"
        yield from (
            doc
            for doc in self._list_all(url)
            if self.settings.POST_TAG_ID not in doc.get("tags", [])
        )

    def download_content(self, doc_id: int) -> tuple[bytes, str]:
        """
        Download the content of a document from Paperless.

        Returns a tuple containing the raw file content as bytes and the content type.
        This method is a pure network operation and does not interact with the filesystem.
        """
        url = f"{self.settings.PAPERLESS_URL}/api/documents/{doc_id}/download/"
        response = self._get(url)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "application/pdf")
        return response.content, content_type

    def update_document(self, doc_id: int, content: str, new_tags: list[int]) -> None:
        """
        Upload content back to Paperless and set the new tags.
        """
        url = f"{self.settings.PAPERLESS_URL}/api/documents/{doc_id}/"
        payload = {"content": content, "tags": new_tags}
        response = self._patch(url, json=payload)
        response.raise_for_status()