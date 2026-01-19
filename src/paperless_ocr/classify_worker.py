"""
Document Classification Worker
==============================

This module defines the ClassificationProcessor, which handles the end-to-end
classification workflow for a document after OCR is complete.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
from typing import Iterable

import structlog

from .classifier import ClassificationProvider, ClassificationResult
from .config import Settings
from .paperless import PaperlessClient

log = structlog.get_logger(__name__)

ERROR_PHRASES = [
    "i'm sorry, i can't assist with that.",
    "i can't assist with that",
    "chatgpt refused to transcribe",
    "[redacted]",
]

PAGE_HEADER_RE = re.compile(r"^--- Page \d+ ---$", re.MULTILINE)
MODEL_FOOTER_RE = re.compile(r"transcribed by model:\s*(.+)", re.IGNORECASE)

MAX_TAXONOMY_CHOICES = 100

COMPANY_SUFFIXES = {
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "gmbh",
    "llc",
    "plc",
    "corp",
    "corporation",
    "co",
    "company",
    "sa",
    "spa",
    "sarl",
    "bv",
    "oy",
    "ab",
    "as",
}

GENERIC_DOCUMENT_TYPES = {
    "document",
    "documents",
    "other",
    "misc",
    "miscellaneous",
    "unknown",
    "general",
    "unspecified",
    "n/a",
    "na",
    "none",
}

BLACKLISTED_TAGS = {
    "new",
    "ai",
    "error",
    "indexed",
}


def _normalize_simple(value: str) -> str:
    """Normalize strings by collapsing whitespace and lowercasing."""
    return " ".join(value.lower().split())


def _normalize_name(value: str) -> str:
    """Normalize organisation names, stripping punctuation and common suffixes."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", value.lower())
    parts = cleaned.split()
    while parts and parts[-1] in COMPANY_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _dedupe_tags(tags: Iterable[str]) -> list[str]:
    """Deduplicate tags while preserving order (case-insensitive)."""
    seen = set()
    output = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(tag)
    return output


def _filter_redundant_tags(
    tags: Iterable[str],
    correspondent: str,
    document_type: str,
    person: str,
) -> list[str]:
    """
    Remove tags that duplicate the correspondent, document type, or person name.
    """
    correspondent_key = _normalize_name(correspondent) if correspondent else ""
    document_type_key = _normalize_simple(document_type) if document_type else ""
    person_key = _normalize_simple(person) if person else ""

    filtered = []
    for tag in _dedupe_tags(tags):
        tag_simple = _normalize_simple(tag)
        tag_name = _normalize_name(tag)
        if correspondent_key and (tag_simple == correspondent_key or tag_name == correspondent_key):
            continue
        if document_type_key and tag_simple == document_type_key:
            continue
        if person_key and tag_simple == person_key:
            continue
        filtered.append(tag)
    return filtered


def _filter_blacklisted_tags(tags: Iterable[str]) -> list[str]:
    """Remove tags that are explicitly blacklisted."""
    filtered = []
    for tag in _dedupe_tags(tags):
        if _normalize_simple(tag) in BLACKLISTED_TAGS:
            continue
        filtered.append(tag)
    return filtered


def _extract_year(value: str) -> str | None:
    """Extract a YYYY year from an ISO-8601 date string."""
    if not value:
        return None
    value = value.strip()
    try:
        return dt.date.fromisoformat(value.split("T")[0]).strftime("%Y")
    except ValueError:
        return None


def _parse_document_date(value: str) -> str | None:
    """Validate and normalize a YYYY-MM-DD date string."""
    if not value:
        return None
    value = value.strip()
    try:
        return dt.date.fromisoformat(value.split("T")[0]).isoformat()
    except ValueError:
        log.warning("Invalid document_date from classifier", value=value)
        return None


def _resolve_date_for_tags(result_date: str | None, existing_date: str | None) -> str:
    """Pick the best available date for year-tag derivation."""
    for value in (result_date, existing_date):
        if not value:
            continue
        try:
            return dt.date.fromisoformat(value.split("T")[0]).isoformat()
        except ValueError:
            continue
    return dt.date.today().isoformat()


def _normalize_language(language: str) -> str | None:
    """Normalize language codes to ISO-639-1 or 'und'."""
    if not language:
        return None
    language = language.strip().lower()
    if language == "und":
        return language
    if len(language) == 2 and language.isalpha():
        return language
    if "-" in language or "_" in language:
        prefix = re.split(r"[-_]", language, maxsplit=1)[0]
        if len(prefix) == 2 and prefix.isalpha():
            return prefix
    return "und"


def _is_generic_document_type(value: str) -> bool:
    """Return True if the document type looks like a generic placeholder."""
    if not value:
        return True
    normalized = _normalize_simple(value)
    return normalized in GENERIC_DOCUMENT_TYPES


def _needs_error_tag(text: str) -> bool:
    """Return True if the OCR text contains refusal markers."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in ERROR_PHRASES)


def _extract_model_tags(text: str) -> list[str]:
    """
    Extract model tags from any "Transcribed by model:" markers in the text.
    """
    matches = list(MODEL_FOOTER_RE.finditer(text))
    if not matches:
        return []

    tags = []
    for match in matches:
        value = match.group(1).splitlines()[0]
        for token in value.split(","):
            tag = token.strip()
            if tag:
                tags.append(tag)
    return _dedupe_tags(tags)


def _split_footer(content: str) -> tuple[str, str]:
    """Split OCR footer to keep model markers when truncating."""
    marker = "\n\nTranscribed by model:"
    index = content.rfind(marker)
    if index == -1:
        return content, ""
    return content[:index], content[index:]


def _extract_page_numbers(matches: list[re.Match]) -> list[int]:
    """Extract page numbers from OCR header matches."""
    numbers = []
    for match in matches:
        raw = match.group(0)
        num_match = re.search(r"\d+", raw)
        if num_match:
            numbers.append(int(num_match.group()))
        else:
            numbers.append(len(numbers) + 1)
    return numbers


def _format_page_ranges(page_numbers: list[int]) -> str:
    """Format a list of page numbers as compact ranges."""
    if not page_numbers:
        return ""
    ordered = sorted(set(page_numbers))
    ranges = []
    start = ordered[0]
    prev = ordered[0]
    for num in ordered[1:]:
        if num == prev + 1:
            prev = num
            continue
        ranges.append((start, prev))
        start = prev = num
    ranges.append((start, prev))
    parts = []
    for start, end in ranges:
        parts.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(parts)


def _page_truncation_note(included_pages: list[int], total_pages: int) -> str:
    """Build a note describing which OCR pages are included."""
    ranges = _format_page_ranges(included_pages)
    return (
        "NOTE: The document transcription below was truncated to reduce cost. "
        f"Included pages (from OCR headers): {ranges}. "
        f"Total pages with OCR headers: {total_pages}."
    )


def _headerless_truncation_note(limit: int) -> str:
    """Build a note describing headerless truncation by character count."""
    return (
        "NOTE: The document transcription below was truncated because page headers "
        f"were not found. Included the first {limit} characters."
    )


def _max_char_truncation_note(limit: int) -> str:
    """Build a note describing extra truncation by max character limit."""
    return (
        "NOTE: The document transcription below was further truncated to the first "
        f"{limit} characters due to the max character limit."
    )


def truncate_content_by_chars(content: str, max_chars: int) -> str:
    """Truncate OCR content by characters, preserving the footer if present."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    body, footer = _split_footer(content)
    if len(body) <= max_chars:
        return body + footer
    return body[:max_chars] + footer


def truncate_content_by_pages(
    content: str,
    max_pages: int,
    tail_pages: int,
    headerless_char_limit: int,
) -> tuple[str, str | None]:
    """
    Truncate OCR content to the first N pages and last N pages using page headers.
    Falls back to a character limit if no page headers are present.
    """
    if max_pages <= 0:
        return content, None

    body, footer = _split_footer(content)
    matches = list(PAGE_HEADER_RE.finditer(body))
    if not matches:
        truncated = truncate_content_by_chars(content, headerless_char_limit)
        if truncated == content:
            return content, None
        return truncated, _headerless_truncation_note(headerless_char_limit)

    page_numbers = _extract_page_numbers(matches)
    total_pages = len(matches)
    if total_pages <= max_pages:
        return body + footer, None

    segments: list[str] = []
    for idx, match in enumerate(matches):
        start = 0 if idx == 0 else match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        segments.append(body[start:end])

    include = set(range(min(max_pages, total_pages)))
    tail_pages = max(0, tail_pages)
    if tail_pages:
        tail_start = max(total_pages - tail_pages, 0)
        include.update(range(tail_start, total_pages))

    include_indices = sorted(include)
    if len(include_indices) >= total_pages:
        return body + footer, None

    truncated_body = "".join(segments[index] for index in include_indices)
    included_pages = [page_numbers[index] for index in include_indices]
    note = _page_truncation_note(included_pages, total_pages)
    return truncated_body + footer, note


def enrich_tags(
    tags: list[str],
    text: str,
    document_date: str,
    default_country_tag: str,
    tag_limit: int,
) -> list[str]:
    """
    Combine model-suggested tags with required tags.
    Required tags are always included and do not count toward tag_limit.
    """
    text_lower = text.lower()
    tag_limit = max(0, tag_limit)

    base_tags = _dedupe_tags(tags)
    required_tags = []

    if any(phrase in text_lower for phrase in ERROR_PHRASES):
        required_tags.append("ERROR")

    required_tags.extend(_extract_model_tags(text))

    year_tag = _extract_year(document_date) or dt.date.today().strftime("%Y")
    required_tags.append(year_tag)

    if default_country_tag:
        required_tags.append(default_country_tag)

    required_tags = _dedupe_tags(required_tags)
    required_set = {tag.lower() for tag in required_tags}

    non_required = [
        tag for tag in base_tags if tag.lower() not in required_set
    ]
    trimmed = non_required[:tag_limit] if tag_limit else []

    if len(non_required) > len(trimmed):
        log.warning(
            "Trimmed non-required tags to limit",
            tag_limit=tag_limit,
            before=len(non_required),
            after=len(trimmed),
        )

    combined = _dedupe_tags(required_tags + trimmed)
    return [tag.lower() for tag in combined]


def _index_items(items: list[dict], normalizer) -> dict[str, dict]:
    """Index Paperless items by normalized name for fast lookup."""
    mapping = {}
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        mapping[normalizer(name)] = item
    return mapping


def _match_item(name: str, mapping: dict[str, dict], normalizer, allow_substring: bool) -> dict | None:
    """Find an existing item by normalized name with optional substring matching."""
    normalized = normalizer(name)
    if not normalized:
        return None
    matched = mapping.get(normalized)
    if matched:
        return matched
    if allow_substring:
        for key, item in mapping.items():
            if normalized in key or key in normalized:
                return item
    return None


def _get_usage_count(item: dict) -> int:
    """Return the usage count for an item if available."""
    for key in ("document_count", "documents_count", "documents"):
        if key not in item:
            continue
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        if isinstance(value, list):
            return len(value)
    return 0


def _top_names(items: list[dict], limit: int) -> list[str]:
    """Return up to limit names sorted by usage count, then name."""
    deduped = {}
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key not in deduped:
            deduped[key] = {"name": name, "count": _get_usage_count(item)}
        else:
            deduped[key]["count"] = max(deduped[key]["count"], _get_usage_count(item))

    sorted_items = sorted(
        deduped.values(),
        key=lambda entry: (-entry["count"], entry["name"].lower()),
    )
    return [entry["name"] for entry in sorted_items[:limit]]


class TaxonomyCache:
    """
    Cache and helper for correspondents, document types, and tags.
    """

    def __init__(self, paperless_client: PaperlessClient):
        self._client = paperless_client
        self._lock = threading.RLock()
        self._correspondents: list[dict] = []
        self._document_types: list[dict] = []
        self._tags: list[dict] = []
        self._correspondent_map: dict[str, dict] = {}
        self._document_type_map: dict[str, dict] = {}
        self._tag_map: dict[str, dict] = {}

    def refresh(self) -> None:
        """Fetch and index the latest taxonomy lists from Paperless."""
        with self._lock:
            self._correspondents = self._client.list_correspondents()
            self._document_types = self._client.list_document_types()
            self._tags = self._client.list_tags()
            self._correspondent_map = _index_items(self._correspondents, _normalize_name)
            self._document_type_map = _index_items(self._document_types, _normalize_simple)
            self._tag_map = _index_items(self._tags, _normalize_simple)

    def correspondent_names(self) -> list[str]:
        """Return correspondent names for prompt context."""
        with self._lock:
            return _top_names(self._correspondents, MAX_TAXONOMY_CHOICES)

    def document_type_names(self) -> list[str]:
        """Return document type names for prompt context."""
        with self._lock:
            return _top_names(self._document_types, MAX_TAXONOMY_CHOICES)

    def tag_names(self) -> list[str]:
        """Return tag names for prompt context."""
        with self._lock:
            return _top_names(self._tags, MAX_TAXONOMY_CHOICES)

    def get_or_create_correspondent_id(self, name: str) -> int | None:
        """Resolve or create a correspondent, returning its ID."""
        if not name.strip():
            return None
        with self._lock:
            matched = _match_item(name, self._correspondent_map, _normalize_name, True)
            if matched:
                return matched.get("id")
            try:
                created = self._client.create_correspondent(name.strip())
            except Exception:
                log.warning("Failed to create correspondent; refreshing cache", name=name)
                self.refresh()
                matched = _match_item(name, self._correspondent_map, _normalize_name, True)
                if matched:
                    return matched.get("id")
                raise
            self._correspondents.append(created)
            self._correspondent_map[_normalize_name(created.get("name", name))] = created
            return created.get("id")

    def get_or_create_document_type_id(self, name: str) -> int | None:
        """Resolve or create a document type, returning its ID."""
        if not name.strip():
            return None
        with self._lock:
            matched = _match_item(name, self._document_type_map, _normalize_simple, False)
            if matched:
                return matched.get("id")
            try:
                created = self._client.create_document_type(name.strip())
            except Exception:
                log.warning("Failed to create document type; refreshing cache", name=name)
                self.refresh()
                matched = _match_item(name, self._document_type_map, _normalize_simple, False)
                if matched:
                    return matched.get("id")
                raise
            self._document_types.append(created)
            self._document_type_map[_normalize_simple(created.get("name", name))] = created
            return created.get("id")

    def get_or_create_tag_ids(self, tags: Iterable[str]) -> list[int]:
        """Resolve or create tags, returning their IDs."""
        matching_algorithm = None
        ids = []
        with self._lock:
            for tag in self._tags:
                value = tag.get("matching_algorithm")
                if isinstance(value, int):
                    matching_algorithm = 0
                    break
                if isinstance(value, str):
                    matching_algorithm = "none"
                    break
        if matching_algorithm is None:
            matching_algorithm = "none"
        for tag in _dedupe_tags(tags):
            with self._lock:
                matched = _match_item(tag, self._tag_map, _normalize_simple, False)
                if matched:
                    ids.append(matched.get("id"))
                    continue
                try:
                    created = self._client.create_tag(
                        tag.strip(), matching_algorithm=matching_algorithm
                    )
                except Exception:
                    log.warning("Failed to create tag; refreshing cache", name=tag)
                    self.refresh()
                    matching_algorithm = None
                    with self._lock:
                        for tag_item in self._tags:
                            value = tag_item.get("matching_algorithm")
                            if isinstance(value, int):
                                matching_algorithm = 0
                                break
                            if isinstance(value, str):
                                matching_algorithm = "none"
                                break
                    if matching_algorithm is None:
                        matching_algorithm = "none"
                    matched = _match_item(tag, self._tag_map, _normalize_simple, False)
                    if matched:
                        ids.append(matched.get("id"))
                        continue
                    raise
                self._tags.append(created)
                self._tag_map[_normalize_simple(created.get("name", tag))] = created
                ids.append(created.get("id"))
        return [tag_id for tag_id in ids if tag_id is not None]


def _update_custom_fields(
    existing: list[dict] | None, field_id: int, value: str
) -> list[dict]:
    """Upsert a Paperless custom field value."""
    existing = existing or []
    updated = []
    found = False
    for item in existing:
        if item.get("field") == field_id:
            updated.append({"field": field_id, "value": value})
            found = True
        else:
            updated.append(item)
    if not found:
        updated.append({"field": field_id, "value": value})
    return updated


def _is_empty_classification(result: ClassificationResult) -> bool:
    """Return True if the classification result contains no usable fields."""
    if result.tags and any(tag.strip() for tag in result.tags):
        return False
    fields = [
        result.title,
        result.correspondent,
        result.document_type,
        result.document_date,
        result.language,
        result.person,
    ]
    return not any(field.strip() for field in fields)


class ClassificationProcessor:
    """
    Orchestrates the classification of a single Paperless document.
    """

    def __init__(
        self,
        doc: dict,
        paperless_client: PaperlessClient,
        classifier: ClassificationProvider,
        taxonomy_cache: TaxonomyCache,
        settings: Settings,
    ):
        self.doc = doc
        self.paperless_client = paperless_client
        self.classifier = classifier
        self.taxonomy_cache = taxonomy_cache
        self.settings = settings
        self.doc_id = doc["id"]
        self.title = doc.get("title") or "<untitled>"

    def process(self) -> None:
        """Run the classification workflow with retries and tag handling."""
        log.info("Classifying document", doc_id=self.doc_id, title=self.title)

        document = self.paperless_client.get_document(self.doc_id)
        content = document.get("content", "") or ""
        current_tags = set(document.get("tags", []))

        if self.settings.ERROR_TAG_ID and self.settings.ERROR_TAG_ID in current_tags:
            log.warning("Document has error tag; skipping classification", doc_id=self.doc_id)
            self._finalize_with_error(current_tags)
            return

        if not content.strip():
            log.warning("Document has no OCR content; requeueing", doc_id=self.doc_id)
            self._requeue_for_ocr(current_tags)
            return

        input_text = content
        truncation_notes: list[str] = []
        if self.settings.CLASSIFY_MAX_PAGES > 0:
            truncated, note = truncate_content_by_pages(
                content,
                self.settings.CLASSIFY_MAX_PAGES,
                self.settings.CLASSIFY_TAIL_PAGES,
                self.settings.CLASSIFY_HEADERLESS_CHAR_LIMIT,
            )
            if truncated != content:
                input_text = truncated
                if note:
                    truncation_notes.append(note)
                log.info(
                    "Truncated document content for classification",
                    doc_id=self.doc_id,
                    max_pages=self.settings.CLASSIFY_MAX_PAGES,
                    tail_pages=self.settings.CLASSIFY_TAIL_PAGES,
                )

        if self.settings.CLASSIFY_MAX_CHARS > 0 and len(input_text) > self.settings.CLASSIFY_MAX_CHARS:
            input_text = input_text[: self.settings.CLASSIFY_MAX_CHARS]
            truncation_notes.append(
                _max_char_truncation_note(self.settings.CLASSIFY_MAX_CHARS)
            )
            log.info(
                "Truncated document content for classification",
                doc_id=self.doc_id,
                max_chars=self.settings.CLASSIFY_MAX_CHARS,
            )

        result, model = self.classifier.classify_text(
            input_text,
            self.taxonomy_cache.correspondent_names(),
            self.taxonomy_cache.document_type_names(),
            self.taxonomy_cache.tag_names(),
            truncation_note="\n".join(truncation_notes) if truncation_notes else None,
        )
        if not result or _is_empty_classification(result):
            log.warning("Classification failed", doc_id=self.doc_id)
            self._finalize_with_error(current_tags)
            return
        if _is_generic_document_type(result.document_type):
            log.warning(
                "Classification returned generic document type",
                doc_id=self.doc_id,
                document_type=result.document_type,
            )
            self._finalize_with_error(current_tags)
            return

        self._apply_classification(document, content, result, model)

    def _clean_processing_tags(self, tags: set[int]) -> set[int]:
        """Remove OCR/classification processing tags and error tags."""
        updated = set(tags)
        updated.discard(self.settings.PRE_TAG_ID)
        updated.discard(self.settings.POST_TAG_ID)
        updated.discard(self.settings.CLASSIFY_PRE_TAG_ID)
        if self.settings.CLASSIFY_POST_TAG_ID:
            updated.discard(self.settings.CLASSIFY_POST_TAG_ID)
        if self.settings.ERROR_TAG_ID:
            updated.discard(self.settings.ERROR_TAG_ID)
        return updated

    def _update_tags(self, tags: set[int]) -> None:
        """Persist tag changes without touching other metadata."""
        self.paperless_client.update_document_metadata(self.doc_id, tags=list(tags))

    def _requeue_for_ocr(self, tags: set[int]) -> None:
        """Move the document back to the OCR queue."""
        updated = self._clean_processing_tags(tags)
        updated.add(self.settings.PRE_TAG_ID)
        self._update_tags(updated)
        log.info("Requeued document for OCR", doc_id=self.doc_id)

    def _finalize_with_error(self, tags: set[int]) -> None:
        """Mark the document with an error tag and clear processing tags."""
        if not self.settings.ERROR_TAG_ID:
            log.warning("Error tag is not configured; skipping error tagging", doc_id=self.doc_id)
            return
        self._update_tags({self.settings.ERROR_TAG_ID})
        log.warning("Marked document as classification error", doc_id=self.doc_id)

    def _apply_classification(
        self,
        document: dict,
        content: str,
        result: ClassificationResult,
        model: str,
    ) -> None:
        """Apply classifier output to Paperless metadata and tags."""
        error_required = _needs_error_tag(content)
        if error_required:
            log.warning(
                "OCR content contains refusal markers; marking error",
                doc_id=self.doc_id,
            )
            self._finalize_with_error(set(document.get("tags", [])))
            return
        parsed_date = _parse_document_date(result.document_date)
        date_for_tags = _resolve_date_for_tags(parsed_date, document.get("created"))
        base_tags = _filter_blacklisted_tags(result.tags)
        base_tags = _filter_redundant_tags(
            base_tags,
            result.correspondent,
            result.document_type,
            result.person,
        )
        tags = enrich_tags(
            base_tags,
            content,
            date_for_tags,
            self.settings.CLASSIFY_DEFAULT_COUNTRY_TAG,
            self.settings.CLASSIFY_TAG_LIMIT,
        )

        if self.settings.ERROR_TAG_ID:
            tags = [tag for tag in tags if tag.strip().lower() != "error"]

        tag_ids = self.taxonomy_cache.get_or_create_tag_ids(tags)
        correspondent_id = (
            self.taxonomy_cache.get_or_create_correspondent_id(result.correspondent)
            if result.correspondent
            else None
        )
        document_type_id = (
            self.taxonomy_cache.get_or_create_document_type_id(result.document_type)
            if result.document_type
            else None
        )

        current_tags = self._clean_processing_tags(set(document.get("tags", [])))
        if self.settings.CLASSIFY_POST_TAG_ID:
            current_tags.add(self.settings.CLASSIFY_POST_TAG_ID)
        current_tags.update(tag_ids)
        if self.settings.ERROR_TAG_ID and error_required:
            current_tags.add(self.settings.ERROR_TAG_ID)

        custom_fields = None
        if self.settings.CLASSIFY_PERSON_FIELD_ID and result.person:
            custom_fields = _update_custom_fields(
                document.get("custom_fields"),
                self.settings.CLASSIFY_PERSON_FIELD_ID,
                result.person,
            )

        language = _normalize_language(result.language)
        title = result.title.strip() if result.title else ""

        self.paperless_client.update_document_metadata(
            self.doc_id,
            title=title or None,
            correspondent_id=correspondent_id,
            document_type_id=document_type_id,
            document_date=parsed_date,
            tags=list(current_tags),
            language=language,
            custom_fields=custom_fields,
        )

        log.info(
            "Document classification applied",
            doc_id=self.doc_id,
            model=model,
            tags_added=len(tag_ids),
        )
