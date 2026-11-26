"""
tests/test_pose_video.py — one-shot video recognition (recognize_video).

The leaderboard input is an mp4 clip with one main performer; these tests run
the temporal path over a synthetic clip with a fake pose engine (the 57-column
row convention of test_pose_plugin.py), so no TensorRT, no network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import vision_stubs  # noqa: F401  (importing installs the ROS stubs)
from vision_stubs import _FakeExecutor

import plugins.pose as pose_plugin  # noqa: E402
from plugins.vision_runtime import LetterboxMeta  # noqa: E402
from test_pose_plugin import _FakeModel, _standing_row  # noqa: E402


def _plugin(cfg=None, model=None):
    plugin = pose_plugin.PosePerceptionPlugin(cfg or {"action_backend": "rules"},
                                              "testns", _FakeExecutor())
    plugin._model = _FakeModel() if model is None else model
    return plugin


def _make_video(path, n_frames=90, fps=30, size=(640, 480)):
    """A real container the decoder must open; frame content is irrelevant —
    the fake engine answers with its preset rows regardless of pixels."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (size[0], size[1]))
    assert writer.isOpened(), "test host cannot write mp4v clips"
    for _ in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), 100, dtype=np.uint8))
    writer.release()
    return str(path)


class _FixedActivityClassifier:
    """Stands in for the hybrid backend: one label, one posture, every call."""

    history_s = 3.0

    def __init__(self, name="clapping", score=0.9):
        self._name = name
        self._score = score
        self.calls = 0

    def classify(self, frames, want_activity=True):
        # Count the learned runs, which is what the throttle bounds: the
        # geometry answer is free and runs every frame, by design.
        if want_activity:
            self.calls += 1
        from plugins.pose_stgcn import ntu_name_zh
        result = {"posture": "standing", "posture_confidence": 1.0,
                  "activity": None, "evidence": {}}
        if want_activity and frames:
            # The real backend answers with the NTU-60 pair, zh included.
            result["activity"] = {"name": self._name,
                                  "name_zh": ntu_name_zh(9),
                                  "score": self._score, "source": "fake"}
        return result


def test_the_reply_carries_the_leaderboard_fields(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=90)
    plugin = _plugin()
    result = plugin.dispatch("pose", {"action": "recognize_video",
                                      "video_path": video})
    assert result["ok"] is True
    for key in ("timestamp", "count", "persons", "latency_ms"):
        assert key in result
    assert result["count"] == 1
    person = result["persons"][0]
    assert person["id"] == 1
    assert isinstance(person["position"], list) and len(person["position"]) == 2
    # The rules read the standing skeleton but a motionless clip has no
    # activity — the person is still reported, the activity fields are not.
    assert person["posture"] == "standing"
    assert "activity" not in person
    assert result["temporal"] is True


def test_a_clip_without_people_is_an_empty_result_not_an_error(tmp_path):
    video = _make_video(tmp_path / "empty.mp4", n_frames=30)
    plugin = _plugin(model=_FakeModel(rows=[]))
    result = plugin.dispatch("pose", {"action": "recognize_video",
                                      "video_path": video})
    assert result["ok"] is True
    assert result["count"] == 0
    assert result["persons"] == []


def test_the_clip_verdict_is_the_dominant_stable_label(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=90)
    plugin = _plugin()
    fixed = _FixedActivityClassifier()
    plugin._classifier_for = lambda merged: fixed
    result = plugin.dispatch("pose", {"action": "recognize_video",
                                      "video_path": video})
    person = result["persons"][0]
    assert person["activity"] == "clapping"
    assert person["activity_zh"] == "鼓掌"
    assert person["activity_score"] == 0.9
    # Throttling held: far fewer model runs than sampled frames.
    assert fixed.calls < result["frames_processed"]


def test_unopenable_bytes_are_reported_not_disguised(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not a video container at all")
    plugin = _plugin()
    result = plugin.dispatch("pose", {"action": "recognize_video",
                                      "video_path": str(bad)})
    assert result["ok"] is False
    assert result["reason"] == "video_unreadable"


def test_missing_or_unrooted_paths_are_bad_input(tmp_path):
    plugin = _plugin()
    missing = plugin.dispatch("pose", {"action": "recognize_video",
                                       "video_path": "/tmp/definitely_absent.mp4"})
    assert missing["ok"] is False and "detail" in missing
    outside = plugin.dispatch("pose", {"action": "recognize_video",
                                       "video_path": "/etc/hostname"})
    assert outside["ok"] is False
    assert "must be under" in outside["detail"]


def test_no_arguments_is_bad_input(tmp_path):
    plugin = _plugin()
    result = plugin.dispatch("pose", {"action": "recognize_video"})
    assert result["ok"] is False
    assert "video_path" in result["detail"]


def test_a_clip_sent_to_the_photo_path_is_rerouted(tmp_path):
    video = _make_video(tmp_path / "clip.mp4", n_frames=60)
    plugin = _plugin()
    result = plugin.dispatch("pose", {"action": "recognize_by_photo",
                                      "image_path": video})
    assert result["ok"] is True
    assert result["temporal"] is True
    assert result["count"] == 1


def test_long_clips_are_sampled_bounded(tmp_path):
    # 30 s at 30 fps: 12 fps sampling would be 360 frames, under the cap;
    # the assertion is that the cap itself is honoured, so push past it.
    video = _make_video(tmp_path / "long.mp4", n_frames=3300, fps=30)
    plugin = _plugin()
    result = plugin.dispatch("pose", {"action": "recognize_video",
                                      "video_path": video})
    assert result["ok"] is True
    assert result["frames_processed"] <= pose_plugin.MAX_VIDEO_FRAMES
    assert result["count"] == 1
