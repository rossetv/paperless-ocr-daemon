# Test Audit Report

**Date:** 2026-03-04
**Project:** paperless-ocr-daemon
**Python:** 3.11+
**Test Framework:** pytest + pytest-cov + pytest-mock + respx

---

## Current Test Suite Summary

| Metric               | Value   |
|----------------------|---------|
| Total tests          | 359     |
| Passed               | 359     |
| Failed               | 0       |
| Skipped              | 0       |
| Flaky                | 0       |
| Execution time       | ~18s    |
| Overall branch cov.  | 99%     |
| Lines missed         | 7/1640  |

## Coverage by Module

| Module                        | Stmts | Miss | Branch | BrPart | Cover |
|-------------------------------|-------|------|--------|--------|-------|
| classifier/__init__.py        | 5     | 0    | 0      | 0      | 100%  |
| classifier/constants.py       | 7     | 0    | 0      | 0      | 100%  |
| classifier/content_prep.py    | 76    | 0    | 30     | 0      | 100%  |
| classifier/daemon.py          | 48    | 1    | 14     | 2      | 95%   |
| classifier/metadata.py        | 55    | 0    | 24     | 1      | 99%   |
| classifier/normalizers.py     | 12    | 0    | 2      | 0      | 100%  |
| classifier/prompts.py         | 4     | 0    | 0      | 0      | 100%  |
| classifier/provider.py        | 114   | 0    | 28     | 0      | 100%  |
| classifier/result.py          | 39    | 0    | 10     | 0      | 100%  |
| classifier/tag_filters.py     | 82    | 0    | 30     | 1      | 99%   |
| classifier/taxonomy.py        | 159   | 0    | 56     | 2      | 99%   |
| classifier/worker.py          | 115   | 0    | 34     | 2      | 99%   |
| common/__init__.py            | 0     | 0    | 0      | 0      | 100%  |
| common/bootstrap.py           | 30    | 0    | 0      | 0      | 100%  |
| common/claims.py              | 35    | 0    | 6      | 0      | 100%  |
| common/concurrency.py         | 20    | 0    | 4      | 0      | 100%  |
| common/config.py              | 131   | 0    | 30     | 0      | 100%  |
| common/daemon_loop.py         | 48    | 0    | 16     | 1      | 98%   |
| common/library_setup.py       | 15    | 0    | 2      | 0      | 100%  |
| common/llm.py                 | 17    | 0    | 4      | 0      | 100%  |
| common/logging_config.py      | 21    | 0    | 4      | 0      | 100%  |
| common/paperless.py           | 137   | 3    | 30     | 0      | 98%   |
| common/preflight.py           | 37    | 0    | 6      | 0      | 100%  |
| common/retry.py               | 34    | 0    | 6      | 1      | 98%   |
| common/shutdown.py            | 19    | 0    | 0      | 0      | 100%  |
| common/stale_lock.py          | 31    | 1    | 8      | 2      | 92%   |
| common/tags.py                | 55    | 0    | 14     | 0      | 100%  |
| common/utils.py               | 13    | 0    | 0      | 0      | 100%  |
| ocr/__init__.py               | 5     | 0    | 0      | 0      | 100%  |
| ocr/daemon.py                 | 44    | 1    | 14     | 2      | 95%   |
| ocr/image_converter.py        | 19    | 0    | 4      | 0      | 100%  |
| ocr/prompts.py                | 2     | 0    | 0      | 0      | 100%  |
| ocr/provider.py               | 67    | 1    | 12     | 0      | 99%   |
| ocr/text_assembly.py          | 22    | 0    | 12     | 0      | 100%  |
| ocr/worker.py                 | 122   | 0    | 30     | 0      | 100%  |

## Uncovered Lines

- `classifier/daemon.py:126` — `if __name__ == "__main__"` guard
- `ocr/daemon.py:133` — `if __name__ == "__main__"` guard
- `ocr/provider.py:69` — Image auto-close in `__del__`
- `common/stale_lock.py:61` — non-integer doc_id skip
- `common/paperless.py:299-301` — `close()` exception swallowing

## Partially Covered Branches

- `classifier/daemon.py:50→58` — post-tag not in tags branch
- `classifier/metadata.py:87→89` — language None early return
- `classifier/tag_filters.py:161→159` — year-tag dedup branch
- `classifier/taxonomy.py:72→71, 296→292` — cache-hit short circuits
- `classifier/worker.py:180→182, 301→303` — empty-result and stats branches
- `common/daemon_loop.py:123→exit` — shutdown-requested log after loop
- `common/retry.py:79→exit` — max retries exceeded exit path

## Test Organization Issues

The existing test suite is **flat** — all 34 test files sit in a single `tests/` directory with no subdirectories. There is no separation between unit, integration, and e2e tests. Some files have `_extended` suffixes that duplicate grouping (e.g., `test_claims.py` + `test_claims_extended.py`).

## Testing Anti-patterns Found

1. **Duplicated test files**: Several modules have a base test file AND an `_extended` file (claims, daemon_loop, classify_worker, classifier_provider, taxonomy, ocr_worker). This creates confusion about where to add new tests.
2. **No test directory structure**: All tests in one flat directory rather than mirroring `src/` layout.
3. **No shared test factories/fixtures**: Test objects (Settings, documents, providers) are re-created from scratch in every test file. Substantial copy-paste of setup code.
4. **No categorization markers**: No pytest marks to distinguish unit vs integration vs e2e tests.
5. **Inconsistent naming**: Some files use the module name (`test_config.py`), others use the feature (`test_classify_e2e.py`).

## Recommended Approach

The existing pytest + respx + pytest-mock stack is well-suited. No framework change needed.

**Recommendations:**
1. Restructure tests to mirror source layout with `tests/unit/`, `tests/integration/`, `tests/e2e/`
2. Merge `_extended` files into single comprehensive test modules
3. Create shared fixtures and factories in `tests/helpers/`
4. Add pytest markers for test categories
5. Fill remaining branch coverage gaps (~7 uncovered lines, ~14 partial branches)
6. Add concurrency stress tests for thread-safe components
