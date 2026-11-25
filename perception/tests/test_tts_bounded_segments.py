"""
tests/test_tts_bounded_segments.py — long-text memory bound on the raw TTS path.

Prior 3-instance platform evaluation OOM'd (exit 137) while synthesizing
800+ char cases: the /tts/test path handed the whole text to a single kokoro
generate(), whose autoregressive activations scale with input length on the
shared-memory Orin. _bounded_segments caps every segment at RAW_SYNTH_MAX_CHARS
(punctuation-aware, hard-split fallback) so the per-call peak is constant.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import pytest

import plugins.tts as tts  # noqa: E402


class TestBoundedSegments:
    def test_empty_and_short(self):
        assert tts._bounded_segments("") == []
        assert tts._bounded_segments("hello") == ["hello"]

    def test_short_text_single_segment(self):
        text = "今天天气不错。"
        assert tts._bounded_segments(text) == [text]

    def test_no_segment_exceeds_cap(self):
        # Punctuation-free CJK longer than the cap must be hard-split.
        text = "字" * (tts.RAW_SYNTH_MAX_CHARS * 3 + 37)
        segs = tts._bounded_segments(text)
        assert all(len(s) <= tts.RAW_SYNTH_MAX_CHARS for s in segs)
        # Content is preserved (order + all characters).
        assert "".join(segs) == text

    def test_cjk_punctuation_boundaries(self):
        text = ("第一段文字。" * 3) + ("第二段文字！" * 3) + ("尾句？" * 5)
        segs = tts._bounded_segments(text)
        assert all(len(s) <= tts.RAW_SYNTH_MAX_CHARS for s in segs)
        assert "".join(segs) == text

    def test_latin_space_hard_split(self):
        words = ("word%d " % (i % 97)) for i in range(400)
        text = "".join(words).strip()
        assert len(text) > tts.RAW_SYNTH_MAX_CHARS
        segs = tts._bounded_segments(text)
        assert all(len(s) <= tts.RAW_SYNTH_MAX_CHARS for s in segs)
        # Spaces may be trimmed at the cut; word characters preserved.
        assert "".join(segs).replace(" ", "") == text.replace(" ", "")

    def test_decimal_not_split_on_period(self):
        text = "圆周率是3.14159，直径10.5厘米。"
        segs = tts._bounded_segments(text)
        assert "".join(segs) == text
        # The decimal must survive inside a single segment.
        joined = segs
        assert any("3.14159" in s for s in joined)

    def test_respects_smaller_cap(self):
        text = "a" * 500
        segs = tts._bounded_segments(text, max_chars=100)
        assert all(len(s) <= 100 for s in segs)
        assert "".join(segs) == text

    def test_800_plus_char_case_bounded(self):
        # The reported OOM trigger: ~800+ char input (this string is 900 chars).
        base = "这是一段用于测试的中文长文本，用于验证内存上界是否生效。"
        text = (base * 40)[:900]
        assert len(text) >= 800
        segs = tts._bounded_segments(text)
        assert len(segs) >= 3
        assert all(len(s) <= tts.RAW_SYNTH_MAX_CHARS for s in segs)
        assert "".join(segs) == text


class TestWorkerSplitDelegates:
    def test_node_split_matches_bounded(self):
        text = "长" * 700 + "。短句。"
        # _split_text is a staticmethod on _TTSNode; call it unbound.
        node_segs = tts._TTSNode._split_text(text, max_chars=280)
        assert node_segs == tts._bounded_segments(text, 280)
        assert all(len(s) <= 280 for s in node_segs)
