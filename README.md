# Paperless-ngx AI OCR Daemon

This repository contains a Python daemon designed to work with [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). It continuously polls a Paperless-ngx instance for documents tagged with a specific "pre-OCR" tag. For each new document, it downloads the file, submits every page to an AI vision model for transcription, and uploads the resulting plain text back to Paperless-ngx, swapping the "pre-OCR" tag for a "post-OCR" tag.

This allows for a powerful, AI-driven OCR workflow that can often produce higher-quality results than traditional OCR engines, especially for documents with complex layouts or handwriting.

## Key Features

*   **AI-Powered Transcription:** Leverages powerful AI vision models (e.g., GPT-4, Gemma) for high-accuracy text extraction.
*   **Modular & Maintainable:** The application is built with a clean, modular architecture, making it easy to understand, maintain, and extend.
*   **Robust Error Handling:** Implements a specific, configurable retry mechanism with exponential backoff for network requests, ensuring resilience against transient errors.
*   **Multi-Page Parallel Processing:** Processes pages of a multi-page document concurrently to improve throughput.
*   **Primary/Fallback Model Support:** Automatically falls back to a secondary model if the primary model refuses to transcribe a page.
*   **Comprehensive Testing:** Includes a full suite of unit tests to ensure code quality and prevent regressions.
*   **Containerized Deployment:** Comes with a multi-stage `Dockerfile` that automates testing and produces a lean, secure production image.
*   **Structured Logging:** Configurable structured (JSON) logging for easy integration with modern log analysis platforms.

## Architecture Overview

The application is broken down into several key components, each with a distinct responsibility. This separation of concerns makes the codebase clean and easy to navigate.

```
+---------------------+      +---------------------+      +--------------------+
|      main.py        |----->|     worker.py       |----->|      ocr.py        |
| (Daemon Entrypoint) |      | (DocumentProcessor) |      |   (OcrProvider)    |
+---------------------+      +---------------------+      +--------------------+
          |                            |
          |                            |
+---------------------+      +---------------------+
|     config.py       |<-----|    paperless.py     |
| (Settings Loading)  |      |  (PaperlessClient)  |
+---------------------+      +---------------------+
```

*   **`main.py`**: The main entry point that starts the daemon, loads configuration, and enters the polling loop.
*   **`worker.py`**: Contains the `DocumentProcessor`, which orchestrates the end-to-end processing of a single document.
*   **`paperless.py`**: The `PaperlessClient` handles all API communication with the Paperless-ngx instance.
*   **`ocr.py`**: The `OcrProvider` abstracts the interaction with the AI model provider (e.g., OpenAI, Ollama).
*   **`config.py`**: Manages loading and validating all configuration from environment variables.
*   **`utils.py`**: Provides common utilities, such as the `@retry` decorator.

## Configuration

The application is configured entirely through environment variables.

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `PAPERLESS_URL` | The full URL of your Paperless-ngx instance. | `http://paperless:8000` | No |
| `PAPERLESS_TOKEN` | Your Paperless-ngx API token. | - | **Yes** |
| `LLM_PROVIDER` | The AI model provider to use. Can be `openai` or `ollama`. | `openai` | No |
| `OPENAI_API_KEY` | Your OpenAI API key. | - | Yes (if `LLM_PROVIDER` is `openai`) |
| `OLLAMA_BASE_URL` | The base URL for your Ollama API instance. | `http://localhost:11434/v1/` | Yes (if `LLM_PROVIDER` is `ollama`) |
| `PRIMARY_MODEL` | The name of the primary AI model to use for transcription. | `gpt-5-mini` (OpenAI) / `gemma3:27b` (Ollama) | No |
| `FALLBACK_MODEL` | The model to use if the primary one fails or refuses. | `o4-mini` (OpenAI) / `gemma3:12b` (Ollama) | No |
| `PRE_TAG_ID` | The ID of the Paperless-ngx tag that marks documents for OCR. | `443` | No |
| `POST_TAG_ID` | The ID of the tag to apply after OCR is complete. | `444` | No |
| `POLL_INTERVAL` | The number of seconds to wait between polling for new documents. | `15` | No |
| `MAX_RETRIES` | The maximum number of times to retry a failed network request. | `20` | No |
| `REQUEST_TIMEOUT` | The timeout in seconds for requests to the AI model provider. | `180` | No |
| `OCR_DPI` | The resolution (in DPI) to use when rasterizing PDF pages. | `300` | No |
| `OCR_MAX_SIDE` | The maximum size (in pixels) of the longest side of an image sent to the model. | `1600` | No |
| `WORKERS` | The number of worker threads to use for parallel page processing. | `8` | No |
| `DOCUMENT_WORKERS` | The number of worker threads to use for parallel document processing. | `4` | No |
| `LOG_LEVEL` | The minimum log level to output. | `INFO` | No |
| `LOG_FORMAT` | The log output format. Can be `console` or `json`. | `console` | No |

## Deployment

The recommended way to deploy the daemon is using the provided `Dockerfile`.

1.  **Build the Docker Image:**
    ```bash
    docker build -t paperless-ocr-daemon .
    ```
    This command will build the image, installing all dependencies and running the test suite to ensure everything is correct.

2.  **Run the Container:**
    Run the container, passing in the required environment variables.
    ```bash
    docker run -d --name ocr-daemon \
      -e PAPERLESS_URL="http://your-paperless-instance" \
      -e PAPERLESS_TOKEN="your_secret_token" \
      -e OPENAI_API_KEY="your_openai_key" \
      -e PRE_TAG_ID="123" \
      -e POST_TAG_ID="456" \
      paperless-ocr-daemon
    ```

## Local Development

To set up a local development environment:

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install the project in editable mode along with the development and testing libraries.
    ```bash
    pip install -e . -r requirements-dev.txt
    ```

4.  **Set Environment Variables:**
    You can create a `.env` file and use a tool like `direnv` or manually export the required variables.

5.  **Run the Tests:**
    To run the full test suite:
    ```bash
    pytest
    ```

6.  **Run the Daemon:**
    ```bash
    python3 -m src.paperless_ocr.main
    ```
