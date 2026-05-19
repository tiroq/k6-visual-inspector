"""Overlay image writing for cluster representatives."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import Cluster, ScreenshotItem
from ..image.layout import rects_to_overlay_image


def write_overlay_images(items: List[ScreenshotItem], clusters: List[Cluster], out_dir: Path) -> None:
    """Write a layout-overlay JPEG for each cluster's representative screenshot."""
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        rep = items[cluster.representative_index]
        out_path = overlay_dir / f"cluster-{cluster.cluster_id:03d}.jpg"
        rects_to_overlay_image(Path(rep.path), rep.rects, out_path)
