"""Tests for orca_teleop.retargeting.calibration."""

import queue

import numpy as np
import pytest
from conftest import plausible_hand_keypoints

from orca_teleop.retargeting.calibration import (
    CalibrationProfile,
    RetargetCalibrationError,
    _emit,
    _prompt,
    default_profile_path,
    load_profile,
    resolve_profile_path,
    run_retarget_calibration,
    save_profile,
)
from orca_teleop.retargeting.constants import CALIBRATION_FRAMES


class _FakeLandmarks:
    def __init__(self, keypoints: np.ndarray) -> None:
        self.keypoints = keypoints
        self.wrist_angle_degrees = 0.0


class _FakeRetargeter:
    """Minimal stand-in matching AdaptiveAnalyticalRetargeter's calibration contract."""

    def __init__(self) -> None:
        self._calibration_done = False
        self._mano_scale = 1.0
        self._frames_ingested = 0

    def reset(self) -> None:
        self._calibration_done = False
        self._mano_scale = 1.0
        self._frames_ingested = 0

    def retarget(self, target_pose):
        if self._calibration_done:
            return "action"
        self._frames_ingested += 1
        if self._frames_ingested >= CALIBRATION_FRAMES:
            self._mano_scale = 1.2345
            self._calibration_done = True
        return None


def _filled_queue(n: int) -> "queue.Queue":
    q: "queue.Queue" = queue.Queue()
    kp = plausible_hand_keypoints()
    for _ in range(n):
        q.put(_FakeLandmarks(kp))
    return q


def test_emit_swallows_callback_exceptions():
    def bad_callback(event):
        raise RuntimeError("boom")

    _emit(bad_callback, "some_event", foo=1)  # must not raise


def test_emit_noop_with_no_callback():
    _emit(None, "some_event")  # must not raise


def test_prompt_raises_without_callback():
    with pytest.raises(RetargetCalibrationError):
        _prompt(None, "confirm_neutral_pose")


def test_prompt_calls_callback_with_action_and_payload():
    seen = []
    _prompt(seen.append, "confirm_neutral_pose", message="hi")
    assert seen == [{"action": "confirm_neutral_pose", "message": "hi"}]


def test_run_retarget_calibration_happy_path():
    retargeter = _FakeRetargeter()
    landmarks_q = _filled_queue(CALIBRATION_FRAMES + 5)
    events = []

    mano_scale = run_retarget_calibration(
        retargeter,
        landmarks_q,
        countdown_s=0.0,
        progress_callback=events.append,
        prompt_callback=lambda p: None,
        should_stop=lambda: False,
    )

    assert mano_scale == pytest.approx(1.2345)
    event_names = [e["event"] for e in events]
    assert event_names[0] == "calibration_started"
    assert event_names[-1] == "calibration_done"
    assert event_names.count("sample_recorded") == CALIBRATION_FRAMES
    assert events[-1]["mano_scale"] == pytest.approx(1.2345)
    assert events[-1]["frames_seen"] == CALIBRATION_FRAMES


def test_run_retarget_calibration_requires_prompt_callback():
    retargeter = _FakeRetargeter()
    landmarks_q = _filled_queue(CALIBRATION_FRAMES)
    with pytest.raises(RetargetCalibrationError):
        run_retarget_calibration(retargeter, landmarks_q, countdown_s=0.0)


def test_run_retarget_calibration_times_out_on_empty_queue():
    retargeter = _FakeRetargeter()
    landmarks_q: "queue.Queue" = queue.Queue()
    events = []

    result = run_retarget_calibration(
        retargeter,
        landmarks_q,
        countdown_s=0.0,
        frame_timeout_s=0.05,
        progress_callback=events.append,
        prompt_callback=lambda p: None,
    )

    assert result is None
    assert events[-1]["event"] == "calibration_timed_out"


def test_run_retarget_calibration_aborts_on_should_stop():
    retargeter = _FakeRetargeter()
    landmarks_q = _filled_queue(CALIBRATION_FRAMES)
    events = []

    result = run_retarget_calibration(
        retargeter,
        landmarks_q,
        countdown_s=0.0,
        progress_callback=events.append,
        prompt_callback=lambda p: None,
        should_stop=lambda: True,
    )

    assert result is None
    assert events[-1]["event"] == "calibration_aborted"


def test_run_retarget_calibration_resets_before_sampling():
    retargeter = _FakeRetargeter()
    retargeter._calibration_done = True
    retargeter._mano_scale = 99.0
    landmarks_q = _filled_queue(CALIBRATION_FRAMES)

    mano_scale = run_retarget_calibration(
        retargeter,
        landmarks_q,
        countdown_s=0.0,
        prompt_callback=lambda p: None,
    )

    assert mano_scale == pytest.approx(1.2345)


def test_profile_round_trip(tmp_path):
    profile = CalibrationProfile(
        operator="eugene",
        mano_scale=1.2345,
        model_path="/some/model",
        urdf_path="/some/urdf",
        calibrated_at="2026-09-21T00:00:00+00:00",
        frames_used=CALIBRATION_FRAMES,
        finger_ratios={"thumb": 1.0, "index": 1.1, "middle": 1.2, "ring": 1.3, "pinky": 1.4},
    )
    path = tmp_path / "eugene.yaml"
    save_profile(path, profile)
    loaded = load_profile(path)
    assert loaded == profile


def test_load_profile_missing_raises_with_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate_retargeting.py"):
        load_profile(tmp_path / "nobody.yaml")


def test_resolve_profile_path_bare_operator_uses_default_dir():
    assert resolve_profile_path("eugene") == default_profile_path("eugene")


def test_resolve_profile_path_explicit_yaml_path_passthrough(tmp_path):
    explicit = tmp_path / "somewhere" / "custom.yaml"
    assert resolve_profile_path(str(explicit)) == explicit
