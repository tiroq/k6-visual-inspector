"""Perceptual hash computation and similarity scoring."""

from __future__ import annotations

import imagehash
from PIL import Image


def compute_hashes(image: Image.Image):
    """Return (phash_str, dhash_str) for *image*."""
    p_hash = imagehash.phash(image)
    d_hash = imagehash.dhash(image)
    return str(p_hash), str(d_hash)


def hamming_hash_similarity(hash_a: str, hash_b: str, hash_bits: int = 64) -> float:
    """Return [0, 1] visual similarity from two hex hash strings."""
    a = imagehash.hex_to_hash(hash_a)
    b = imagehash.hex_to_hash(hash_b)
    distance = a - b
    return max(0.0, 1.0 - (distance / hash_bits))
