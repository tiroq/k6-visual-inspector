"""Tests for config.resolve_workers."""

import pytest

from k6_visual_inspector.config import resolve_workers


def test_auto_returns_positive_int():
    result = resolve_workers("auto")
    assert isinstance(result, int)
    assert result >= 1


def test_auto_safe_returns_one():
    assert resolve_workers("auto-safe") == 1


def test_auto_max_returns_positive_int():
    result = resolve_workers("auto-max")
    assert isinstance(result, int)
    assert result >= 1


def test_integer_string():
    assert resolve_workers("4") == 4


def test_integer_one():
    assert resolve_workers("1") == 1


def test_invalid_string_raises_system_exit():
    with pytest.raises(SystemExit):
        resolve_workers("invalid")


def test_zero_raises_system_exit():
    with pytest.raises(SystemExit):
        resolve_workers("0")


def test_negative_raises_system_exit():
    with pytest.raises(SystemExit):
        resolve_workers("-1")
