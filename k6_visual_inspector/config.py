"""AppConfig dataclass and worker-count resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    input_dir: Path
    output_dir: Path

    ocr_engine: str = "tesseract"
    ocr_lang: str = "eng"
    ocr_mode: str = "fast"

    threshold: float = 0.82
    template_threshold: float = 0.84
    final_threshold: float = 0.78
    single_stage: bool = False

    visual_weight: float = 0.30
    text_weight: float = 0.25
    layout_weight: float = 0.25
    rule_weight: float = 0.20

    workers: str = "auto"
    copy: bool = False
    limit: int = 0
    debug_ocr: bool = False

    # Resolved at runtime — not a CLI argument.
    debug_ocr_dir: Optional[Path] = field(default=None, repr=False)


def resolve_workers(value: str) -> int:
    """Return the number of worker processes for the given configuration string.

    Supported values:
      auto       — half of logical CPU cores (conservative default)
      auto-safe  — 1 worker (always safe, no multiprocessing overhead)
      auto-max   — all logical CPU cores
      <integer>  — exact number, must be >= 1
    """
    cpu_count = os.cpu_count() or 1

    if value == "auto":
        return max(1, cpu_count // 2)

    if value == "auto-safe":
        return 1

    if value == "auto-max":
        return max(1, cpu_count)

    try:
        workers = int(value)
    except ValueError:
        raise SystemExit(
            f"Invalid --workers value: {value!r}. "
            "Use an integer or one of: auto, auto-safe, auto-max."
        )

    if workers < 1:
        raise SystemExit("--workers must be >= 1")

    return workers
