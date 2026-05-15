"""Agglomerative clustering on precomputed similarity matrices."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..models import ScreenshotItem


def cluster_by_matrix(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> List[int]:
    """Cluster items using agglomerative clustering on a precomputed similarity matrix."""
    n = similarity_matrix.shape[0]

    if n == 0:
        return []

    if n == 1:
        return [0]

    distance_matrix = 1.0 - similarity_matrix
    distance_threshold = 1.0 - threshold

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    return [int(x) for x in model.fit_predict(distance_matrix)]


def cluster_items_single_stage(
    items: List[ScreenshotItem],
    combined_similarity_matrix: np.ndarray,
    threshold: float,
) -> List[int]:
    """Single-stage clustering using the combined similarity matrix."""
    return cluster_by_matrix(combined_similarity_matrix, threshold=threshold)


def two_stage_cluster_items(
    items: List[ScreenshotItem],
    visual_matrix: np.ndarray,
    layout_matrix: np.ndarray,
    text_matrix: np.ndarray,
    rule_matrix: np.ndarray,
    template_threshold: float,
    final_threshold: float,
) -> List[int]:
    """Two-stage clustering: first group by visual/layout template, then by text/symptoms."""
    n = len(items)

    template_matrix = (
        visual_matrix * 0.45
        + layout_matrix * 0.45
        + rule_matrix * 0.10
    )

    template_labels = cluster_by_matrix(template_matrix, threshold=template_threshold)

    for item, label in zip(items, template_labels):
        item.template_cluster_id = label

    template_to_indices: Dict[int, List[int]] = {}

    for idx, label in enumerate(template_labels):
        template_to_indices.setdefault(label, []).append(idx)

    final_labels = [-1] * n
    next_cluster_id = 0

    for _, indices in sorted(template_to_indices.items(), key=lambda kv: min(kv[1])):
        if len(indices) == 1:
            final_labels[indices[0]] = next_cluster_id
            next_cluster_id += 1
            continue

        sub_matrix = np.eye(len(indices), dtype=np.float32)

        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                if local_i >= local_j:
                    continue

                score = (
                    visual_matrix[global_i, global_j] * 0.20
                    + layout_matrix[global_i, global_j] * 0.20
                    + text_matrix[global_i, global_j] * 0.35
                    + rule_matrix[global_i, global_j] * 0.25
                )

                sub_matrix[local_i, local_j] = score
                sub_matrix[local_j, local_i] = score

        sub_labels = cluster_by_matrix(sub_matrix, threshold=final_threshold)
        local_to_global_cluster: Dict[int, int] = {}

        for local_idx, sub_label in enumerate(sub_labels):
            if sub_label not in local_to_global_cluster:
                local_to_global_cluster[sub_label] = next_cluster_id
                next_cluster_id += 1

            global_idx = indices[local_idx]
            final_labels[global_idx] = local_to_global_cluster[sub_label]

    for item, label in zip(items, final_labels):
        item.final_cluster_id = label

    return final_labels
