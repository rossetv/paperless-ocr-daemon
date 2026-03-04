# Test Architecture

## Directory Layout

```
tests/
├── conftest.py                         # Root config: sys.path, shared fixtures, pytest markers
├── helpers/
│   ├── __init__.py
│   ├── factories.py                    # Test object factories (Settings, documents, results)
│   └── mocks.py                        # Reusable mock builders for PaperlessClient, providers
├── unit/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── test_bootstrap.py           # common.bootstrap
│   │   ├── test_claims.py              # common.claims
│   │   ├── test_concurrency.py         # common.concurrency
│   │   ├── test_config.py              # common.config
│   │   ├── test_daemon_loop.py         # common.daemon_loop
│   │   ├── test_library_setup.py       # common.library_setup
│   │   ├── test_llm.py                 # common.llm
│   │   ├── test_logging_config.py      # common.logging_config
│   │   ├── test_paperless.py           # common.paperless
│   │   ├── test_preflight.py           # common.preflight
│   │   ├── test_retry.py              # common.retry
│   │   ├── test_shutdown.py            # common.shutdown
│   │   ├── test_stale_lock.py          # common.stale_lock
│   │   ├── test_tags.py               # common.tags
│   │   └── test_utils.py              # common.utils
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── test_constants.py           # classifier.constants
│   │   ├── test_content_prep.py        # classifier.content_prep
│   │   ├── test_daemon.py              # classifier.daemon
│   │   ├── test_metadata.py            # classifier.metadata
│   │   ├── test_normalizers.py         # classifier.normalizers
│   │   ├── test_provider.py            # classifier.provider
│   │   ├── test_result.py              # classifier.result
│   │   ├── test_tag_filters.py         # classifier.tag_filters
│   │   ├── test_taxonomy.py            # classifier.taxonomy
│   │   └── test_worker.py             # classifier.worker
│   └── ocr/
│       ├── __init__.py
│       ├── test_daemon.py              # ocr.daemon
│       ├── test_image_converter.py     # ocr.image_converter
│       ├── test_provider.py            # ocr.provider
│       ├── test_text_assembly.py       # ocr.text_assembly
│       └── test_worker.py             # ocr.worker
├── integration/
│   ├── __init__.py
│   ├── test_ocr_pipeline.py            # OCR download → convert → transcribe → upload
│   └── test_classifier_pipeline.py     # Classify fetch → truncate → LLM → apply metadata
└── e2e/
    ├── __init__.py
    ├── test_ocr_workflow.py            # Full OCR daemon document lifecycle
    └── test_classifier_workflow.py     # Full classification daemon document lifecycle
```

## Naming Conventions

- Test files: `test_<module_name>.py` — one file per source module
- Test functions: `test_<function>_<scenario>_<expected>` (e.g., `test_parse_date_empty_string_returns_none`)
- Factories: `make_<entity>(overrides)` (e.g., `make_settings()`, `make_document()`)
- Fixtures: descriptive nouns (e.g., `settings`, `paperless_client`, `mock_provider`)

## How to Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# E2E tests only
pytest tests/e2e/

# Single module
pytest tests/unit/common/test_config.py

# Single test
pytest tests/unit/common/test_config.py::test_settings_default_values

# With coverage
pytest --cov=src --cov-report=term-missing --cov-branch

# Randomized order (if pytest-randomly installed)
pytest -p randomly
```

## Fixtures and Factories

### Factories (`tests/helpers/factories.py`)

Factories produce valid test objects with sensible defaults:

```python
make_settings(**overrides)     # Returns a Settings with all required fields populated
make_document(**overrides)     # Returns a Paperless document dict
make_classification_result(**) # Returns a ClassificationResult
```

### Mocks (`tests/helpers/mocks.py`)

Reusable mock builders:

```python
make_mock_paperless(**overrides)  # Returns a MagicMock PaperlessClient
make_mock_ocr_provider(**)       # Returns a MagicMock OcrProvider
make_mock_classify_provider(**)  # Returns a MagicMock ClassificationProvider
```

## Pytest Markers

```ini
[tool.pytest.ini_options]
markers =
    unit: Unit tests (fast, no I/O)
    integration: Integration tests (module boundaries)
    e2e: End-to-end tests (full workflows)
```

Tests are auto-marked by directory via `conftest.py` — no manual marking needed.
