"""
Pytest configuration.

Why this exists:

The project uses a ``src/`` layout (package code lives in ``src/common``,
``src/ocr``, and ``src/classifier``).
Normally, developers run tests after installing the package (e.g. ``pip install -e .``).

On some macOS/Python 3.13 setups, editable installs in dot-prefixed virtualenv
folders (like ``.venv``) can result in the generated ``.pth`` file being marked
as hidden, and Python's ``site`` module will skip hidden ``.pth`` files. When
that happens, imports fail even though the source tree is
present.

This file makes tests robust in that scenario by adding ``src/`` to ``sys.path``
so the local packages take precedence.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_ensure_src_on_path()
