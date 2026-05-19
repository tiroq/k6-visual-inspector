"""JSON, JSONL, and CSV serialization helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..models import Cluster, ScreenshotItem


def item_to_json_dict(item: ScreenshotItem) -> Dict[str, Any]:
    data = asdict(item)
    data["rects"] = [asdict(r) for r in item.rects]
    return data


def cluster_to_json_dict(cluster: Cluster) -> Dict[str, Any]:
    return asdict(cluster)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_clusters_json(path: Path, clusters: List[Cluster]) -> None:
    path.write_text(
        json.dumps([cluster_to_json_dict(c) for c in clusters], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_cluster_summary_csv(clusters: List[Cluster], out_dir: Path) -> None:
    path = out_dir / "cluster-summary.csv"

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "cluster_name",
                "count",
                "severity",
                "visual_class",
                "text_class",
                "avg_combined_similarity",
                "avg_visual_similarity",
                "avg_text_similarity",
                "avg_layout_similarity",
                "avg_rule_similarity",
                "representative_path",
            ],
        )

        writer.writeheader()

        for c in clusters:
            writer.writerow(
                {
                    "cluster_id": c.cluster_id,
                    "cluster_name": c.cluster_name,
                    "count": c.count,
                    "severity": c.severity,
                    "visual_class": c.visual_class,
                    "text_class": c.text_class,
                    "avg_combined_similarity": round(c.avg_combined_similarity, 4),
                    "avg_visual_similarity": round(c.avg_visual_similarity, 4),
                    "avg_text_similarity": round(c.avg_text_similarity, 4),
                    "avg_layout_similarity": round(c.avg_layout_similarity, 4),
                    "avg_rule_similarity": round(c.avg_rule_similarity, 4),
                    "representative_path": c.representative_path,
                }
            )
