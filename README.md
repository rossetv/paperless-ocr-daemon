# Paperless-ngx AI OCR and Classification Daemons

This project provides two small Python daemons for Paperless-ngx:

- OCR daemon: downloads documents tagged for OCR, transcribes them with an AI vision model, uploads the text, and updates tags.
- Classification daemon: reads OCR text, uses an LLM to classify documents, and updates Paperless metadata and tags.

Docker image: https://hub.docker.com/repository/docker/rossetv/paperless-ocr-daemon

## Pipeline overview

Tag-driven flow (defaults shown):

1) Paperless ingests a document and assigns the inbox tag `PRE_TAG_ID` (default `443`).
2) OCR daemon processes documents with `PRE_TAG_ID`, uploads text, removes `PRE_TAG_ID`, and adds `POST_TAG_ID` (default `444`).
3) Classification daemon processes documents with `CLASSIFY_PRE_TAG_ID` (defaults to `POST_TAG_ID`), updates metadata, then removes the pipeline queue/processing tags.

On success, the queue tags are removed and the document is enriched with the classifier's tags/metadata.
On failure (OCR refusal, redaction markers, invalid classification, etc.), the document is marked with `ERROR_TAG_ID` and removed from the queue.

Non-pipeline (user) tags are preserved in both success and error paths.

If a document somehow ends up with both a queue tag and its corresponding post-tag, the daemons will remove the stale queue tag to keep the pipeline moving.

## Quick start (Docker)

OCR daemon (OpenAI):

```bash
docker run -d --name paperless-ocr-daemon \
  -e PAPERLESS_URL="http://your-paperless-instance" \
  -e PAPERLESS_TOKEN="your_paperless_api_token" \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e PRE_TAG_ID="443" \
  -e POST_TAG_ID="444" \
  rossetv/paperless-ocr-daemon:latest
```

Classification daemon (same image, different command):

```bash
docker run -d --name paperless-classifier-daemon \
  -e PAPERLESS_URL="http://your-paperless-instance" \
  -e PAPERLESS_TOKEN="your_paperless_api_token" \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e CLASSIFY_PRE_TAG_ID="444" \
  -e CLASSIFY_DEFAULT_COUNTRY_TAG="Ireland" \
  rossetv/paperless-ocr-daemon:latest \
  paperless-classifier-daemon
```

You can also run the classifier with `python3 -m src.classifier.daemon` if you prefer.

If you use Ollama instead of OpenAI:

```bash
docker run -d --name paperless-ocr-daemon \
  -e PAPERLESS_URL="http://your-paperless-instance" \
  -e PAPERLESS_TOKEN="your_paperless_api_token" \
  -e LLM_PROVIDER="ollama" \
  -e OLLAMA_BASE_URL="http://ollama:11434/v1/" \
  -e PRE_TAG_ID="443" \
  -e POST_TAG_ID="444" \
  rossetv/paperless-ocr-daemon:latest
```

## Configuration

All configuration is via environment variables.

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `PAPERLESS_URL` | URL of your Paperless-ngx instance. | `http://paperless:8000` | No |
| `PAPERLESS_TOKEN` | Paperless-ngx API token. | - | Yes |
| `LLM_PROVIDER` | AI provider: `openai` or `ollama`. | `openai` | No |
| `OPENAI_API_KEY` | OpenAI API key. | - | Yes (if `LLM_PROVIDER=openai`) |
| `OLLAMA_BASE_URL` | Ollama API base URL. | `http://localhost:11434/v1/` | Yes (if `LLM_PROVIDER=ollama`) |
| `AI_MODELS` | Comma-separated model list used for OCR + classification. | `gpt-5-mini,gpt-5.2,o4-mini` / `gemma3:27b,gemma3:12b` | No |
| `OCR_REFUSAL_MARKERS` | Comma-separated OCR refusal phrases (case-insensitive). | `i can't assist, i cannot assist, i can't help with transcrib, i cannot help with transcrib, CHATGPT REFUSED TO TRANSCRIBE` | No |
| `OCR_INCLUDE_PAGE_MODELS` | Include the model name in each OCR page header. | `false` | No |
| `PRE_TAG_ID` | Tag ID for documents to OCR. | `443` | No |
| `POST_TAG_ID` | Tag ID to apply after OCR. | `444` | No |
| `OCR_PROCESSING_TAG_ID` | Optional tag ID used as an OCR in-progress lock. | - | No |
| `POLL_INTERVAL` | Seconds between polls. | `15` | No |
| `MAX_RETRIES` | Max retries for network/model calls. | `20` | No |
| `MAX_RETRY_BACKOFF_SECONDS` | Max seconds to sleep between retries. | `30` | No |
| `REQUEST_TIMEOUT` | Timeout in seconds for model calls. | `180` | No |
| `OCR_DPI` | DPI when rasterizing PDFs. | `300` | No |
| `OCR_MAX_SIDE` | Max pixel size of the longest image side. | `1600` | No |
| `PAGE_WORKERS` | Parallel page workers per document. | `8` | No |
| `DOCUMENT_WORKERS` | Parallel documents at a time. | `4` | No |
| `LOG_LEVEL` | Minimum log level. | `INFO` | No |
| `LOG_FORMAT` | `console` or `json`. | `console` | No |
| `CLASSIFY_PRE_TAG_ID` | Tag ID for documents to classify. | `POST_TAG_ID` | No |
| `CLASSIFY_POST_TAG_ID` | Optional tag ID to apply after classification. | - | No |
| `CLASSIFY_PROCESSING_TAG_ID` | Optional tag ID used as a classification in-progress lock. | - | No |
| `ERROR_TAG_ID` | Tag ID for OCR/classification error marker. | `552` | No |
| `CLASSIFY_PERSON_FIELD_ID` | Paperless custom field ID for "Person". | - | No |
| `CLASSIFY_DEFAULT_COUNTRY_TAG` | Country tag to always add. | - | No |
| `CLASSIFY_MAX_CHARS` | Max OCR characters to send to classifier (0 = no limit). | `0` | No |
| `CLASSIFY_MAX_TOKENS` | Max output tokens for classification (0 = provider default). | `0` | No |
| `CLASSIFY_TAG_LIMIT` | Max number of non-required tags to keep after enrichment. | `5` | No |
| `CLASSIFY_TAXONOMY_LIMIT` | Max correspondents/types/tags passed into the prompt (0 = no limit). | `100` | No |
| `CLASSIFY_MAX_PAGES` | Max OCR pages to send to classifier (0 = no limit). | `3` | No |
| `CLASSIFY_TAIL_PAGES` | Extra OCR pages from the end to include when truncating. | `2` | No |
| `CLASSIFY_HEADERLESS_CHAR_LIMIT` | Char limit used when OCR page headers are missing. | `15000` | No |

Notes:
- `CLASSIFY_PERSON_FIELD_ID` should point to a Paperless custom field of type "text" used for the person name.
- By default no post-classification tag is added; the pipeline tags are removed instead.

## OCR output format

- Each OCR page is separated with a header like `--- Page 2 ---` when the document has multiple pages.
- A footer lists the models used for transcription.
- Blank pages are skipped.
- If all models fail or refuse, a refusal marker is inserted.
- If OCR produces no text at all (e.g. all pages are blank), the document is marked with `ERROR_TAG_ID` to avoid a requeue loop.
- If a page OCR call raises an unexpected exception, an `[OCR ERROR]` marker is inserted and the document is marked with `ERROR_TAG_ID`.
- If `OCR_INCLUDE_PAGE_MODELS=true`, page headers include the model name (e.g., `--- Page 2 (gpt-5.2) ---`).
- If `OCR_PROCESSING_TAG_ID` is set, the tag is added while OCR runs and removed when finished.

## Classification behavior

- If OCR content is empty, the classifier removes the OCR tag and re-adds the inbox tag so OCR runs again.
- Classification relies on `MAX_RETRIES` in the API client; if all models fail or return empty output, the document is marked with `ERROR_TAG_ID` and pipeline tags are removed.
- If a document already has `ERROR_TAG_ID`, classification is skipped and pipeline tags are removed.
- Page truncation uses the first `CLASSIFY_MAX_PAGES` pages plus the last `CLASSIFY_TAIL_PAGES` pages. If no page headers are present, it falls back to `CLASSIFY_HEADERLESS_CHAR_LIMIT`.
- Required tags (year, country, model markers) are always included and do not count toward `CLASSIFY_TAG_LIMIT`.
- The prompt includes up to `CLASSIFY_TAXONOMY_LIMIT` correspondents, document types, and tags, sorted by usage.
- If `CLASSIFY_PROCESSING_TAG_ID` is set, the tag is added while classification runs and removed when finished.

## Architecture

The codebase is split into reusable top-level packages:

- `src/common/` - shared building blocks (config, Paperless client, retry, logging, daemon loop).
- `src/ocr/` - OCR provider + single-document worker + daemon entrypoint.
- `src/classifier/` - classification provider + single-document worker + daemon entrypoint.

Key modules:

- `src/ocr/daemon.py` - OCR daemon entrypoint.
- `src/ocr/worker.py` - OCR pipeline (single-document worker).
- `src/ocr/provider.py` - OCR provider interface and OpenAI/Ollama implementation.
- `src/classifier/daemon.py` - classification daemon entrypoint.
- `src/classifier/worker.py` - classification pipeline (single-document worker).
- `src/classifier/provider.py` - classification prompt, response parsing, and LLM calls.
- `src/common/paperless.py` - Paperless-ngx API client.
- `src/common/config.py` - configuration and validation.
- `src/common/utils.py` - retry helper utilities.
- `src/common/logging_config.py` - logging setup.
- `src/common/daemon_loop.py` - shared polling loop used by both daemons.

## Runtime dependencies

- Python 3.11+
- Poppler (required for PDF conversion when not using Docker)

## Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e . -r requirements-dev.txt
pytest
python3 -m src.ocr.daemon
```

## Logging

Logs are structured via `structlog`. Use `LOG_FORMAT=json` for machine parsing and `LOG_LEVEL` to control verbosity.

## Privacy and data handling

OCR sends page images to the configured model provider, and classification sends OCR text. Keep API tokens in environment variables and be mindful of where your data is processed.
