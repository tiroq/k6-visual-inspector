"""Tests for analysis.rules — detect_rule_labels."""

from k6_visual_inspector.analysis.rules import detect_rule_labels
from k6_visual_inspector.models import Rect


def _no_rects():
    return []


def _modal_rect():
    return [Rect(x=100, y=100, w=400, h=300, area_ratio=0.25, aspect_ratio=1.33, kind="modal_or_dialog")]


class TestDetectRuleLabels:
    def _call(self, normalized_text="", central_text="", region_texts=None, rects=None):
        return detect_rule_labels(
            normalized_text=normalized_text,
            central_text=central_text,
            region_texts=region_texts or [],
            rects=rects or _no_rects(),
            width=1280,
            height=720,
        )

    def test_internal_server_error_becomes_backend_500(self):
        result = self._call(normalized_text="Internal Server Error")
        assert result["text_class"] == "backend_500"
        assert result["severity"] == "high"

    def test_gateway_timeout_becomes_timeout(self):
        result = self._call(normalized_text="Gateway Timeout occurred")
        assert result["text_class"] == "timeout"
        assert result["severity"] == "high"

    def test_unauthorized_becomes_unauthorized(self):
        result = self._call(normalized_text="401 Unauthorized access denied")
        assert result["text_class"] == "unauthorized"

    def test_forbidden(self):
        result = self._call(normalized_text="403 Forbidden")
        assert result["text_class"] == "forbidden"

    def test_not_found(self):
        result = self._call(normalized_text="404 Not Found")
        assert result["text_class"] == "not_found"
        assert result["severity"] == "medium"

    def test_empty_state_text(self):
        result = self._call(normalized_text="No data available")
        assert result["text_class"] == "empty_state"
        assert result["severity"] == "low"

    def test_loading_state(self):
        result = self._call(normalized_text="Loading please wait")
        assert result["text_class"] == "loading_or_spinner"

    def test_modal_with_error_text_becomes_error_modal(self):
        result = self._call(
            normalized_text="An error occurred please try again",
            rects=_modal_rect(),
        )
        assert result["visual_class"] == "error_modal"

    def test_modal_without_error_text(self):
        result = self._call(
            normalized_text="Are you sure you want to proceed",
            rects=_modal_rect(),
        )
        assert result["visual_class"] == "modal"

    def test_empty_text_no_rects(self):
        result = self._call()
        assert result["visual_class"] == "unknown"
        assert result["text_class"] == "no_text"

    def test_cluster_hint_format(self):
        result = self._call(normalized_text="Internal Server Error")
        assert "." in result["cluster_hint"]
        assert result["cluster_hint"] == f"{result['visual_class']}.{result['text_class']}"

    def test_bad_gateway(self):
        result = self._call(normalized_text="502 Bad Gateway")
        assert result["text_class"] == "bad_gateway_502"

    def test_service_unavailable(self):
        result = self._call(normalized_text="503 Service Unavailable")
        assert result["text_class"] == "service_unavailable_503"

    def test_rate_limit(self):
        result = self._call(normalized_text="429 Too Many Requests rate limit exceeded")
        assert result["text_class"] == "rate_limit"
