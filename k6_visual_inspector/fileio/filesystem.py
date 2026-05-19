"""Filesystem helpers — safe path names, copy/symlink, etc."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


def safe_name(value: str) -> str:
    """Return a filesystem-safe lowercase slug for *value*."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "unknown"


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool) -> None:
    """Copy *src* to *dst*, or create a symlink depending on *use_symlink*."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)
