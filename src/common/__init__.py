"""
Common building blocks shared by the OCR and classification daemons.

This package contains reusable, domain-agnostic code:

- configuration loading (environment variables)
- Paperless-ngx API client
- retry/backoff helpers
- a small polling + threadpool daemon loop
- logging configuration
- small helpers for processing-tag claiming and OpenAI-compatible calls
"""

