"""Tests for the HandTeleopCapture phone-frame adapter."""

import numpy as np

from orca_teleop.ingress.phone.landmarks import PHONE_JOINT_NAMES, landmarks_from_phone_frame


def _joint(name: str, position, confidence: float = 1.0):
    return {
        "name": name,
        "x": 0.5,
        "y": 0.5,
        "confidence": confidence,
        "position": position,
    }


def _complete_frame() -> dict:
    joints = [
        _joint(name, position={"x": float(i), "y": float(i) * 2, "z": float(i) * 3})
        for i, name in enumerate(PHONE_JOINT_NAMES)
    ]
    return {"joints": joints, "confidence": 1.0, "chirality": "right"}


def test_complete_frame_returns_ordered_array():
    frame = _complete_frame()

    keypoints = landmarks_from_phone_frame(frame)

    assert keypoints is not None
    assert keypoints.shape == (21, 3)
    assert keypoints.dtype == np.float32
    np.testing.assert_array_equal(keypoints[0], [0.0, 0.0, 0.0])  # wrist
    np.testing.assert_array_equal(keypoints[5], [5.0, 10.0, 15.0])  # indexMCP


def test_missing_joint_drops_the_frame():
    frame = _complete_frame()
    frame["joints"] = [j for j in frame["joints"] if j["name"] != "thumbCMC"]

    assert landmarks_from_phone_frame(frame) is None


def test_low_confidence_joint_drops_the_frame():
    frame = _complete_frame()
    frame["joints"][4]["confidence"] = 0.01  # thumbTip

    assert landmarks_from_phone_frame(frame, min_confidence=0.5) is None


def test_missing_position_drops_the_frame():
    frame = _complete_frame()
    frame["joints"][8]["position"] = None  # indexTip: no depth reading

    assert landmarks_from_phone_frame(frame) is None


def test_confidence_gate_is_configurable():
    frame = _complete_frame()
    frame["joints"][0]["confidence"] = 0.2  # wrist

    assert landmarks_from_phone_frame(frame, min_confidence=0.5) is None
    assert landmarks_from_phone_frame(frame, min_confidence=0.1) is not None
