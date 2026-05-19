"""Screenshot discovery — find supported image files in a directory tree."""

from __future__ import annotations

from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def find_images(input_dir: Path) -> List[Path]:
    """Return all supported image files under *input_dir*, sorted by path."""
    files: List[Path] = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)
