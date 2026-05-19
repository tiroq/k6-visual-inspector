"""Cluster directory generation — one subfolder per cluster with images."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import Cluster, ScreenshotItem
from ..fileio.filesystem import copy_or_symlink


def create_cluster_directories(
    items: List[ScreenshotItem],
    clusters: List[Cluster],
    out_dir: Path,
    use_symlink: bool,
) -> None:
    """Create one sub-directory per cluster and populate it with screenshots."""
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        cluster_name = f"cluster-{cluster.cluster_id:03d}__{cluster.cluster_name}__count-{cluster.count}"
        cluster_dir = clusters_dir / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for rank, idx in enumerate(cluster.items):
            item = items[idx]
            src = Path(item.path)
            prefix = "representative" if idx == cluster.representative_index else f"item-{rank:03d}"
            dst = cluster_dir / f"{prefix}__{src.name}"
            copy_or_symlink(src, dst, use_symlink=use_symlink)
