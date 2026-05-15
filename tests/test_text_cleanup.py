"""Tests for ocr.cleanup — normalize_text and tokenize_text."""

from k6_visual_inspector.ocr.cleanup import normalize_text, tokenize_text


class TestNormalizeText:
    def test_timestamps_are_normalized(self):
        result = normalize_text("Event at 2024-01-15T12:34:56Z")
        assert "<datetime>" in result
        assert "2024" not in result

    def test_dates_are_normalized(self):
        result = normalize_text("Created on 2023-07-04")
        assert "<date>" in result
        assert "2023" not in result

    def test_uuids_are_normalized(self):
        result = normalize_text("Request ID: 550e8400-e29b-41d4-a716-446655440000")
        assert "<uuid>" in result
        assert "550e8400" not in result

    def test_numbers_are_normalized(self):
        result = normalize_text("Found 42 items and 3.14 ratio")
        assert "<num>" in result
        assert "42" not in result

    def test_emails_are_normalized(self):
        result = normalize_text("Contact support@example.com for help")
        assert "<email>" in result
        assert "support@example.com" not in result

    def test_urls_are_normalized(self):
        result = normalize_text("Visit https://example.com/path?q=1 for details")
        assert "<url>" in result
        assert "https://example.com" not in result

    def test_hex_ids_are_normalized(self):
        result = normalize_text("Hash: a1b2c3d4e5f6a7b8")
        assert "<hex>" in result

    def test_lowercased(self):
        result = normalize_text("HELLO WORLD")
        assert result == result.lower()

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_plain_text_preserved(self):
        result = normalize_text("internal server error")
        assert "internal" in result
        assert "server" in result
        assert "error" in result


class TestTokenizeText:
    def test_basic_tokens(self):
        tokens = tokenize_text("internal server error")
        assert "internal" in tokens
        assert "server" in tokens
        assert "error" in tokens

    def test_stopwords_removed(self):
        tokens = tokenize_text("the quick and the brown fox")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens

    def test_short_tokens_excluded(self):
        tokens = tokenize_text("a bb ccc dddd")
        assert "a" not in tokens  # single char excluded
        assert "bb" in tokens     # 2-char tokens are kept (pattern is {2,})
        assert "ccc" in tokens
        assert "dddd" in tokens

    def test_empty_string(self):
        assert tokenize_text("") == []
