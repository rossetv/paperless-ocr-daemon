"""
Common building blocks shared by the OCR and classification daemons.

This package contains reusable, domain-agnostic code:

- **config** — environment-variable-based settings and library setup.
- **paperless** — Paperless-ngx REST API client with automatic retries.
- **utils** — retry decorator, blank-image detection, redaction-marker check.
- **daemon_loop** — polling + threadpool loop used by both daemon entry points.
- **logging_config** — structured logging (structlog) configuration.
- **llm** — OpenAI-compatible chat completion mixin with retries.
- **claims** — processing-lock tag claiming with race-condition mitigation.
- **tags** — tag extraction, hygiene, and pipeline-tag lifecycle helpers.
"""
