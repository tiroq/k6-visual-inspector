"""Tests for image.layout.iou_rect."""

import pytest

from k6_visual_inspector.models import Rect
from k6_visual_inspector.image.layout import iou_rect


def _rect(x, y, w, h) -> Rect:
    return Rect(x=x, y=y, w=w, h=h, area_ratio=0.1, aspect_ratio=1.0, kind="region")


def test_iou_non_overlapping():
    a = _rect(0, 0, 10, 10)
    b = _rect(20, 20, 10, 10)
    assert iou_rect(a, b) == 0.0


def test_iou_identical():
    a = _rect(5, 5, 20, 20)
    b = _rect(5, 5, 20, 20)
    assert iou_rect(a, b) == pytest.approx(1.0)


def test_iou_partial_overlap():
    a = _rect(0, 0, 10, 10)
    b = _rect(5, 0, 10, 10)
    result = iou_rect(a, b)
    assert 0.0 < result < 1.0


def test_iou_containment():
    outer = _rect(0, 0, 20, 20)
    inner = _rect(5, 5, 10, 10)
    result = iou_rect(outer, inner)
    # Intersection = 10*10 = 100, union = 400 + 100 - 100 = 400
    assert result == pytest.approx(100 / 400)


def test_iou_symmetry():
    a = _rect(0, 0, 15, 15)
    b = _rect(10, 10, 15, 15)
    assert iou_rect(a, b) == pytest.approx(iou_rect(b, a))


def test_iou_zero_area():
    a = _rect(0, 0, 0, 0)
    b = _rect(0, 0, 10, 10)
    assert iou_rect(a, b) == 0.0
