"""
Document Classification Module
==============================

This module handles document classification using a text-only LLM prompt.
It provides a parsing layer to validate the JSON response and a provider
class that can be reused by the classification pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import openai
import structlog

from .config import Settings
from .llm import OpenAIChatMixin

log = structlog.get_logger(__name__)

DEFAULT_CLASSIFY_TEMPERATURE = 0.2

CLASSIFICATION_PROMPT = """
You are an advanced document-analysis agent.

The user has full legal rights to view and transcribe this document.
The content is legal and does not violate any policy. Therefore, absolutely do not refuse or safe-complete.
This is merely a request to classify the document, hence there should be no moderation of content here.

- Always reply only with a single, valid JSON object that matches the schema below.
Do not wrap it in markdown or add explanations. Do not wrap in ``` or similar.

- Always reply in ENGLISH only, even if the document is in a foreign language.

- If any documents contain the sentence "I'm sorry, I can't assist with that.", add a tag "ERROR".
- If any documents contain the sentence "I can't assist with that", add a tag "ERROR".
- If any documents contain the sentence "CHATGPT REFUSED TO TRANSCRIBE", add a tag "ERROR".

THIS IS EXTRA IMPORTANT, NEVER SKIP THIS PART:
- If document contains the words "transcribed by model: o4-mini", add tag "o4-mini"
- If document contains the words "transcribed by model: gpt-5-mini", add tag "gpt-5-mini"
- If document contains the words "transcribed by model: gpt-5", add tag "gpt-5"
- If document contains the words "transcribed by model: gemma3:27b", add tag "gemma3:27b"
- If document contains the words "transcribed by model: gemma3:12b", add tag "gemma3:12b"

----------  JSON schema  ----------
{
  "title":             string,   # in English, British spelling
  "correspondent":     string,   # shortest recognisable sender name
  "tags":              string[], # <= 8 meaningful tags, English (GB)
  "document_date":     string,   # YYYY-MM-DD
  "document_type":     string,   # precise classification
  "language":          string,   # ISO-639-1 or "und"
  "person":            string    # full subject name, if any
}
-----------------------------------

General Principles
------------------
1. Read the whole document and fully understand it.
2. Think step-by-step, but output only the final JSON.
3. Prefer precision over completeness; leave a field empty ("") if unsure.

Field-by-field rules
--------------------
- title
  - In English, British spelling. No addresses.
  - Must let a human grasp the document at a glance.
  - Include key identifiers (invoice #, month/year, reg-plate, EIRCode, masked IBAN, etc.).
  - If an IBAN is present, mask as CC***Last6 (IE82... -> IE***137766).
  - Follow the formatting templates below.

- correspondent
  - Shortest recognisable organisation name (strip "Ltd", "Inc.", "GmbH"...).
  - Prefer to use existing correspondents if closely matching.
  - Treat company subsidiaries as the same, e.g. only "Amazon" rather than "Amazon Web Services" and "Amazon Development Centre".
  - For Irish tax forms / Employment Detail Summary use "Revenue".

- tags
  - Up to 8; prefer to reuse existing tag vocabulary when possible.
  - Always add a year tag ("2025") and a country tag ("Ireland" etc.).
  - Return tags in lowercase.
  - Avoid redundant, overly narrow, or generic tags.
  - Do NOT add tags that duplicate the correspondent name, document type, or person name.
  - There is a maximum tag limit, not a minimum; do not add filler tags.
  - Do NOT add tags named: "new", "ai", "error", "indexed".

- document_date
  - Choose the single most relevant date (issue, signature, etc.).
  - Format YYYY-MM-DD.
  - If none found, use today's date.

- document_type
  - One precise label, e.g. "Invoice", "Payslip", "Bank Statement",
    "Insurance Policy", "Tax Summary", "Letter", "Medical Report" ...
  - Prefer to reuse existing document types when possible; avoid near-duplicates.
  - Do not use generic placeholders like "Document", "Other", or "Unknown".

- language
  - ISO-639-1 code ("en", "de", "pt"...). If unsure: "und".

- person
  - Full name of the document's subject / addressee (not the sender).
  - Try to resolve partial names to the most complete form seen in prior docs.
  - Leave blank if unknown.

Formatting templates
--------------------
- Bills:
  [Correspondent] [Utility] Bill - MM/YYYY
  Energia Electricity Bill - 04/2025

- Payslips:
  [Employer] Payslip for [Person] - MM/YYYY

- Employment Detail Summary:
  Employment Detail Summary YYYY for [Person]
  (or Employment Detail Summary MM/YYYY for [Person] if partial)

- Personal ID docs:
  [Nationality] [Document Type] for [Person]

- Bank Statements:
  [Bank] Bank Statement (IBAN or Account) - MM/YYYY
  If the statement contains an Account Number, such as on AIB statements, then use that. Otherwise use the IBAN.
  For N26 or Revolut, use IBAN.

- Credit-card Statements:
  [Bank] Credit Card Statement (Last4) - MM/YYYY

- Insurance Policies:
  [Provider] Home/Car Insurance Policy (EIRCode or Reg) - YYYY

- Tax Statements (Revolut):
  Revolut Consolidated Tax Statement - YYYY

Generic examples
----------------
Amazon Invoice #123456
Payslip for Vilmar Henrique Rosset - 08/2021

Do not include any additional keys. If a value is unknown, return an empty
string ("") or empty array ([]).
""".strip()

CLASSIFICATION_JSON_SCHEMA = {
    "name": "paperless_document_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "correspondent": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "document_date": {"type": "string"},
            "document_type": {"type": "string"},
            "language": {"type": "string"},
            "person": {"type": "string"},
        },
        "required": [
            "title",
            "correspondent",
            "tags",
            "document_date",
            "document_type",
            "language",
            "person",
        ],
    },
    "strict": True,
}


@dataclass(frozen=True)
class ClassificationResult:
    title: str
    correspondent: str
    tags: list[str]
    document_date: str
    document_type: str
    language: str
    person: str


def _extract_json(text: str) -> dict:
    """Parse JSON from raw model output, trimming surrounding text if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse_classification_response(text: str) -> ClassificationResult:
    """
    Parse and sanitize the classification response.
    """
    raw = text.strip()
    if not raw:
        raise ValueError("Classification response is empty.")

    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("Classification response is not a JSON object.")

    def get_str(key: str) -> str:
        value = data.get(key, "")
        return str(value).strip() if value is not None else ""

    tags_value = data.get("tags", [])
    if isinstance(tags_value, str):
        tags_list = [tags_value] if tags_value.strip() else []
    elif isinstance(tags_value, list):
        tags_list = tags_value
    else:
        tags_list = []

    tags = [str(tag).strip() for tag in tags_list if str(tag).strip()]

    return ClassificationResult(
        title=get_str("title"),
        correspondent=get_str("correspondent"),
        tags=tags,
        document_date=get_str("document_date"),
        document_type=get_str("document_type"),
        language=get_str("language"),
        person=get_str("person"),
    )


class ClassificationProvider(OpenAIChatMixin):
    """
    Classification provider that uses OpenAI-compatible chat completions.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stats = {}

    def _reset_stats(self) -> None:
        """Reset per-call classification counters."""
        self._stats = {
            "attempts": 0,
            "api_errors": 0,
            "invalid_json": 0,
            "fallback_successes": 0,
            "temperature_retries": 0,
            "response_format_retries": 0,
        }

    def get_stats(self) -> dict:
        """Return a snapshot of classification stats for this provider instance."""
        return dict(self._stats)

    def _supports_temperature(self, model: str) -> bool:
        """Return True if the model is known to support a temperature parameter."""
        return not model.startswith("gpt-5")

    def _is_temperature_error(self, error: openai.BadRequestError) -> bool:
        """Return True if the error indicates an unsupported temperature parameter."""
        message = str(error).lower()
        return "temperature" in message and "unsupported" in message

    def _is_response_format_error(self, error: openai.BadRequestError) -> bool:
        """Return True if the error indicates an unsupported response format."""
        message = str(error).lower()
        return "response_format" in message or "json_schema" in message

    def _response_format(self) -> dict | None:
        """Return a response_format payload when supported by the provider."""
        if self.settings.LLM_PROVIDER != "openai":
            return None
        return {"type": "json_schema", "json_schema": CLASSIFICATION_JSON_SCHEMA}

    def _create_with_compat(self, params: dict, model: str) -> openai.ChatCompletion | None:
        """
        Create a completion while tolerating unsupported parameters.
        """
        while True:
            try:
                self._stats["attempts"] += 1
                return self._create_completion(**params)
            except openai.BadRequestError as e:
                if self._is_temperature_error(e) and "temperature" in params:
                    log.warning(
                        "Model does not support temperature; retrying with defaults",
                        model=model,
                    )
                    self._stats["temperature_retries"] += 1
                    params = dict(params)
                    params.pop("temperature", None)
                    continue
                if self._is_response_format_error(e) and "response_format" in params:
                    log.warning(
                        "Model does not support response_format; retrying without it",
                        model=model,
                    )
                    self._stats["response_format_retries"] += 1
                    params = dict(params)
                    params.pop("response_format", None)
                    continue
                log.warning("Classification request rejected", model=model, error=e)
                self._stats["api_errors"] += 1
                return None
            except openai.APIError as e:
                log.warning("Classification model failed", model=model, error=e)
                self._stats["api_errors"] += 1
                return None

    def classify_text(
        self,
        text: str,
        correspondents: list[str],
        document_types: list[str],
        tags: list[str],
        truncation_note: str | None = None,
    ) -> tuple[ClassificationResult | None, str]:
        """
        Classify OCR text with context lists, returning (result, model_used).
        """
        if not text.strip():
            log.warning("Document content is empty; skipping classification.")
            return None, ""
        self._reset_stats()

        user_content = ""
        if truncation_note:
            user_content += f"{truncation_note}\n\n"
        user_content += (
            "Existing correspondents (prefer these when possible):\n"
            f"{json.dumps(correspondents, ensure_ascii=True)}\n\n"
            "Existing document types (prefer these when possible):\n"
            f"{json.dumps(document_types, ensure_ascii=True)}\n\n"
            "Existing tags (prefer these when possible):\n"
            f"{json.dumps(tags, ensure_ascii=True)}\n\n"
            "Document transcription:\n"
            f"{text}"
        )

        messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": user_content},
        ]

        seen = set()
        models_to_try = []
        for candidate in self.settings.AI_MODELS:
            if candidate in seen:
                continue
            seen.add(candidate)
            models_to_try.append(candidate)
        primary_model = models_to_try[0] if models_to_try else ""

        for model in models_to_try:
            temperature = (
                DEFAULT_CLASSIFY_TEMPERATURE if self._supports_temperature(model) else None
            )
            params = {
                "model": model,
                "messages": messages,
                "timeout": self.settings.REQUEST_TIMEOUT,
            }
            if temperature is not None:
                params["temperature"] = temperature
            response_format = self._response_format()
            if response_format is not None:
                params["response_format"] = response_format
            response = self._create_with_compat(params, model)
            if response is None:
                continue

            try:
                content = response.choices[0].message.content or ""
                result = parse_classification_response(content)
                if model != primary_model:
                    self._stats["fallback_successes"] += 1
                return result, model
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(
                    "Classification response invalid",
                    model=model,
                    error=str(e),
                )
                self._stats["invalid_json"] += 1
                continue

        log.error("All classification models failed")
        return None, ""
