# Paperless-ngx AI OCR Daemon

This repository contains a small Python daemon that connects to Paperless-ngx, finds documents tagged for OCR, transcribes them with an AI vision model, and writes the text back to Paperless-ngx while swapping tags.

Docker image: https://hub.docker.com/repository/docker/rossetv/paperless-ocr-daemon

## What this does

- Watches Paperless-ngx for documents with a "pre-OCR" tag.
- Downloads each document, renders pages, and runs OCR.
- Uploads the combined text and replaces the tag with a "post-OCR" tag.

## How it works (short version)

1) The daemon loads settings from environment variables.
2) It polls the Paperless-ngx API for documents with `PRE_TAG_ID`.
3) Each document is processed in parallel (up to `DOCUMENT_WORKERS`).
4) Pages are processed in parallel per document (up to `PAGE_WORKERS`).
5) The resulting text is uploaded and tags are updated.

## How it works (detailed)

- `main.py` creates the settings, configures logging, and enters the poll loop.
- `PaperlessClient` lists documents by tag, downloads files, and updates content.
- `DocumentProcessor` is the pipeline for a single document:
  1) Download the file.
  2) Write it to a temporary file.
  3) Convert the file into images (PDFs via `pdf2image`, images via `Pillow`).
  4) OCR each page with the configured model provider.
  5) Assemble the final text and upload it.
- The "pre-OCR" tag is removed and the "post-OCR" tag is added. Existing tags are preserved.

## Output format

- For multi-page documents, each page is separated with a header like `--- Page 2 ---`.
- A footer lists the models used for transcription.
- Blank pages are skipped.
- If all models fail or refuse, the refusal marker is inserted.

## Architecture

Source modules and responsibilities:

- `src/paperless_ocr/main.py` - daemon entrypoint, polling loop, document-level concurrency.
- `src/paperless_ocr/worker.py` - document processing pipeline.
- `src/paperless_ocr/paperless.py` - Paperless-ngx API client and pagination.
- `src/paperless_ocr/ocr.py` - OCR provider interface and OpenAI/Ollama implementation.
- `src/paperless_ocr/config.py` - settings loading and validation.
- `src/paperless_ocr/utils.py` - retry decorator and helper utilities.
- `src/paperless_ocr/logging_config.py` - structured logging setup.

## Quick start (Docker)

```bash
docker run -d --name paperless-ocr-daemon \
  -e PAPERLESS_URL="http://your-paperless-instance" \
  -e PAPERLESS_TOKEN="your_paperless_api_token" \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e PRE_TAG_ID="123" \
  -e POST_TAG_ID="456" \
  rossetv/paperless-ocr-daemon:latest
```

If you use Ollama instead of OpenAI:

```bash
docker run -d --name paperless-ocr-daemon \
  -e PAPERLESS_URL="http://your-paperless-instance" \
  -e PAPERLESS_TOKEN="your_paperless_api_token" \
  -e LLM_PROVIDER="ollama" \
  -e OLLAMA_BASE_URL="http://ollama:11434/v1/" \
  -e PRE_TAG_ID="123" \
  -e POST_TAG_ID="456" \
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
| `PRIMARY_MODEL` | Primary model name. | `gpt-5-mini` / `gemma3:27b` | No |
| `FALLBACK_MODEL` | Fallback model name. | `o4-mini` / `gemma3:12b` | No |
| `PRE_TAG_ID` | Tag ID for documents to process. | `443` | No |
| `POST_TAG_ID` | Tag ID to apply after OCR. | `444` | No |
| `POLL_INTERVAL` | Seconds between polls. | `15` | No |
| `MAX_RETRIES` | Max retries for network/model calls. | `20` | No |
| `REQUEST_TIMEOUT` | Timeout in seconds for model calls. | `180` | No |
| `OCR_DPI` | DPI when rasterizing PDFs. | `300` | No |
| `OCR_MAX_SIDE` | Max pixel size of the longest image side. | `1600` | No |
| `PAGE_WORKERS` | Parallel page workers per document. | `8` | No |
| `DOCUMENT_WORKERS` | Parallel documents at a time. | `4` | No |
| `LOG_LEVEL` | Minimum log level. | `INFO` | No |
| `LOG_FORMAT` | `console` or `json`. | `console` | No |

## Runtime dependencies

- Python 3.11+
- Poppler (required for PDF conversion when not using Docker)

## Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e . -r requirements-dev.txt
pytest
python3 -m src.paperless_ocr.main
```

## Logging

Logs are structured via `structlog`. Use `LOG_FORMAT=json` for machine parsing and `LOG_LEVEL` to control verbosity.

## Extending the project

Common maintenance tasks:

- Change polling behavior: `src/paperless_ocr/main.py`
- Update OCR prompt or providers: `src/paperless_ocr/ocr.py`
- Adjust document assembly or tag behavior: `src/paperless_ocr/worker.py`
- Add new configuration values: `src/paperless_ocr/config.py`

## Privacy and data handling

OCR requires sending page images to the selected model provider. Make sure you are comfortable with where your data is processed and keep API tokens in environment variables.
