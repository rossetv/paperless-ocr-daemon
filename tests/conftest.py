"""
Pytest configuration.

Why this exists:

The project uses a ``src/`` layout (package code lives in ``src/paperless_ocr``).
Normally, developers run tests after installing the package (e.g. ``pip install -e .``).

On some macOS/Python 3.13 setups, editable installs in dot-prefixed virtualenv
folders (like ``.venv``) can result in the generated ``.pth`` file being marked
as hidden, and Python's ``site`` module will skip hidden ``.pth`` files. When
that happens, ``import paperless_ocr`` fails even though the source tree is
present.

This file makes tests robust in that scenario by adding ``src/`` to ``sys.path``
only when the package cannot be imported normally.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    try:
        import paperless_ocr  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

