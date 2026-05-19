"""CLI argument parsing and main pipeline orchestration."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .config import AppConfig, resolve_workers
from .models import ScreenshotItem
from .fileio.discovery import find_images
from .fileio.serialization import (
    item_to_json_dict,
    write_jsonl,
    write_clusters_json,
    write_cluster_summary_csv,
)
from .clustering.similarity import compute_similarity_matrices
from .clustering.clustering import cluster_items_single_stage, two_stage_cluster_items
from .clustering.cluster_building import build_clusters
from .report.html import generate_html_report
from .report.overlays import write_overlay_images
from .report.cluster_dirs import create_cluster_directories
from .analysis.analyzer import analyze_screenshot_worker


# ---------------------------------------------------------------------------
# Analysis pipeline helpers
# ---------------------------------------------------------------------------

def _analyze_images(
    image_paths: List[Path],
    ocr_lang: str,
    ocr_engine: str,
    ocr_mode: str,
    workers: int,
    debug_ocr_dir: Optional[Path],
) -> List[ScreenshotItem]:
    """Run per-screenshot analysis, returning only successfully processed items."""
    items: List[ScreenshotItem] = []

    tasks = [
        (
            str(path),
            index,
            ocr_lang,
            ocr_engine,
            ocr_mode,
            str(debug_ocr_dir) if debug_ocr_dir is not None else None,
        )
        for index, path in enumerate(image_paths)
    ]

    if workers == 1:
        for task in tqdm(tasks, desc="Analyzing screenshots"):
            path_str = task[0]
            try:
                item = analyze_screenshot_worker(task)
                items.append(item)
            except Exception as e:
                print(f"Failed to analyze {path_str}: {e}")
    else:
        print(f"Using parallel screenshot analysis workers: {workers}")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(analyze_screenshot_worker, task): task
                for task in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing screenshots"):
                task = futures[future]
                path_str = task[0]

                try:
                    item = future.result()
                    items.append(item)
                except Exception as e:
                    print(f"Failed to analyze {path_str}: {e}")

        # Restore deterministic item order after parallel execution.
        items.sort(key=lambda item: item.index)

    return items


def _compact_reindex_items(items: List[ScreenshotItem]) -> None:
    """Reindex items so that indices are contiguous after failures drop some."""
    items.sort(key=lambda item: item.index)
    for new_index, item in enumerate(items):
        item.index = new_index


def _write_outputs(
    out_dir: Path,
    items: List[ScreenshotItem],
    clusters,
    visual_matrix: np.ndarray,
    text_matrix: np.ndarray,
    layout_matrix: np.ndarray,
    rule_matrix: np.ndarray,
    combined_matrix: np.ndarray,
    config: AppConfig,
) -> None:
    write_jsonl(out_dir / "items.jsonl", (item_to_json_dict(item) for item in items))
    write_clusters_json(out_dir / "clusters.json", clusters)
    write_cluster_summary_csv(clusters, out_dir)

    np.save(out_dir / "similarity_visual.npy", visual_matrix)
    np.save(out_dir / "similarity_text.npy", text_matrix)
    np.save(out_dir / "similarity_layout.npy", layout_matrix)
    np.save(out_dir / "similarity_rule.npy", rule_matrix)
    np.save(out_dir / "similarity_combined.npy", combined_matrix)

    write_overlay_images(items, clusters, out_dir)

    create_cluster_directories(
        items=items,
        clusters=clusters,
        out_dir=out_dir,
        use_symlink=not config.copy,
    )

    generate_html_report(
        items=items,
        clusters=clusters,
        out_dir=out_dir,
        visual_weight=config.visual_weight,
        text_weight=config.text_weight,
        layout_weight=config.layout_weight,
        rule_weight=config.rule_weight,
        threshold=config.threshold,
        template_threshold=config.template_threshold,
        final_threshold=config.final_threshold,
        single_stage=config.single_stage,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: AppConfig) -> None:
    """Execute the full screenshot clustering pipeline."""
    if not config.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {config.input_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(config.input_dir)

    if config.limit and config.limit > 0:
        image_paths = image_paths[: config.limit]

    if not image_paths:
        raise SystemExit(f"No supported images found in: {config.input_dir}")

    print(f"Found screenshots: {len(image_paths)}")
    print(f"Output directory: {config.output_dir}")
    print(f"OCR engine: {config.ocr_engine}")
    print(f"OCR language: {config.ocr_lang}")

    workers = resolve_workers(config.workers)
    print(f"Workers: {workers}")

    if workers > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if config.ocr_engine == "easyocr" and workers > 1:
        print(
            "WARNING: EasyOCR loads a heavy torch model per worker. "
            "For EasyOCR, workers=1 is usually safer.\n"
            "         Use multiprocessing primarily with --ocr-engine tesseract."
        )

    debug_ocr_dir = config.output_dir / "debug-ocr" if config.debug_ocr else None
    config.debug_ocr_dir = debug_ocr_dir

    items = _analyze_images(
        image_paths=image_paths,
        ocr_lang=config.ocr_lang,
        ocr_engine=config.ocr_engine,
        ocr_mode=config.ocr_mode,
        workers=workers,
        debug_ocr_dir=debug_ocr_dir,
    )

    if not items:
        raise SystemExit("No screenshots were successfully analyzed.")

    _compact_reindex_items(items)

    visual_matrix, text_matrix, layout_matrix, rule_matrix, combined_matrix = (
        compute_similarity_matrices(
            items,
            visual_weight=config.visual_weight,
            text_weight=config.text_weight,
            layout_weight=config.layout_weight,
            rule_weight=config.rule_weight,
        )
    )

    if config.single_stage:
        labels = cluster_items_single_stage(
            items,
            combined_similarity_matrix=combined_matrix,
            threshold=config.threshold,
        )
    else:
        labels = two_stage_cluster_items(
            items=items,
            visual_matrix=visual_matrix,
            layout_matrix=layout_matrix,
            text_matrix=text_matrix,
            rule_matrix=rule_matrix,
            template_threshold=config.template_threshold,
            final_threshold=config.final_threshold,
        )

    clusters = build_clusters(
        items=items,
        labels=labels,
        visual_matrix=visual_matrix,
        text_matrix=text_matrix,
        layout_matrix=layout_matrix,
        rule_matrix=rule_matrix,
        combined_matrix=combined_matrix,
    )

    clusters.sort(key=lambda c: (-c.count, c.cluster_id))

    _write_outputs(
        out_dir=config.output_dir,
        items=items,
        clusters=clusters,
        visual_matrix=visual_matrix,
        text_matrix=text_matrix,
        layout_matrix=layout_matrix,
        rule_matrix=rule_matrix,
        combined_matrix=combined_matrix,
        config=config,
    )

    print()
    print("Done.")
    print(f"Total screenshots: {len(items)}")
    print(f"Total clusters: {len(clusters)}")
    print(f"HTML report: {config.output_dir / 'report.html'}")
    print(f"Clusters JSON: {config.output_dir / 'clusters.json'}")
    print(f"Items JSONL: {config.output_dir / 'items.jsonl'}")
    print(f"Cluster CSV: {config.output_dir / 'cluster-summary.csv'}")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster UI screenshots by visual, OCR text, layout, and rule-based semantic similarity."
    )

    parser.add_argument(
        "input_dir",
        help="Directory with screenshots.",
    )

    parser.add_argument(
        "output_dir",
        help="Directory where reports and clusters will be written.",
    )

    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "easyocr"],
        default="tesseract",
        help="OCR engine. Default: tesseract.",
    )

    parser.add_argument(
        "--ocr-mode",
        choices=["fast", "balanced", "accurate"],
        default="fast",
        help="OCR speed/quality mode. Default: fast.",
    )

    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="OCR language, e.g. eng, rus, eng+rus. Default: eng.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Combined similarity threshold for single-stage clustering. Default: 0.82.",
    )

    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.84,
        help="Similarity threshold for first-stage visual/layout template clustering. Default: 0.84.",
    )

    parser.add_argument(
        "--final-threshold",
        type=float,
        default=0.78,
        help="Similarity threshold for second-stage text/symptom clustering inside templates. Default: 0.78.",
    )

    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage clustering instead of two-stage clustering.",
    )

    parser.add_argument(
        "--visual-weight",
        type=float,
        default=0.30,
        help="Weight of visual hash similarity. Default: 0.30.",
    )

    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.25,
        help="Weight of OCR text similarity. Default: 0.25.",
    )

    parser.add_argument(
        "--layout-weight",
        type=float,
        default=0.25,
        help="Weight of layout/composition similarity. Default: 0.25.",
    )

    parser.add_argument(
        "--rule-weight",
        type=float,
        default=0.20,
        help="Weight of rule-based semantic similarity. Default: 0.20.",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images into cluster folders instead of creating symlinks.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of screenshots for debugging. 0 means no limit.",
    )

    parser.add_argument(
        "--workers",
        default="auto",
        help=(
            "Number of parallel screenshot analysis workers. "
            "Use 'auto' for half of logical CPU cores, 'auto-safe' for 1, "
            "'auto-max' for all cores, or an integer. Default: auto."
        ),
    )

    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Save OCR debug crops.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry-point used by the CLI and the compatibility wrapper."""
    args = _parse_args()

    config = AppConfig(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        ocr_engine=args.ocr_engine,
        ocr_lang=args.ocr_lang,
        ocr_mode=args.ocr_mode,
        threshold=args.threshold,
        template_threshold=args.template_threshold,
        final_threshold=args.final_threshold,
        single_stage=args.single_stage,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight,
        layout_weight=args.layout_weight,
        rule_weight=args.rule_weight,
        workers=args.workers,
        copy=args.copy,
        limit=args.limit,
        debug_ocr=args.debug_ocr,
    )

    run(config)
