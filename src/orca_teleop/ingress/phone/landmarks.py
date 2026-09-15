"""Adapt HandTeleopCapture's phone JSON frames to the retargeter's (21, 3) layout.

The phone (`teleop/HandTeleopCapture`) already reports 21 MediaPipe-order
landmarks whose `position` field is documented as "metric wrist-relative 3D
position in meters" (see `HandFrame.swift`) -- exactly the convention
`orca_teleop`'s own `hand_stream.proto` documents for `HandFrame.keypoints`.
So unlike the Meta Quest WebXR adapter (25 joints, a different basis, needs a
reduction *and* a basis change), this one does no geometry at all: it looks
up each of the 21 named joints and returns their `position`, or drops the
frame if any of them isn't usable.

`orca_teleop` does not import anything from the `teleop` submodule at
runtime -- ``PHONE_JOINT_NAMES`` below is a second, independent copy of the
wire contract documented in `HandFrame.swift`'s `orderedJoints`, the same
"duplicate a short, stable list rather than round-trip the schema at
runtime" tradeoff `teleop`'s own `stream_common.py` makes for its Python
tools.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Mirrors HandFrame.orderedJoints in teleop/HandTeleopCapture/Data/HandFrame.swift --
# same 21 landmarks, same order. This is already MediaPipe's own landmark
# order, which is what the retargeter expects, so no reindexing table is
# needed here (contrast orca_teleop.ingress.metaquest.landmarks, which does
# need one for WebXR's 25-joint layout).
PHONE_JOINT_NAMES = (
    "wrist",
    "thumbCMC",
    "thumbMCP",
    "thumbIP",
    "thumbTip",
    "indexMCP",
    "indexPIP",
    "indexDIP",
    "indexTip",
    "middleMCP",
    "middlePIP",
    "middleDIP",
    "middleTip",
    "ringMCP",
    "ringPIP",
    "ringDIP",
    "ringTip",
    "littleMCP",
    "littlePIP",
    "littleDIP",
    "littleTip",
)

# Starting point, not a measured constant -- tune against the real pipeline
# on the Intel Mac (see worklog.md) rather than trusting any particular
# number a priori.
DEFAULT_MIN_CONFIDENCE = 0.5


def landmarks_from_phone_frame(
    frame: dict[str, Any],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> np.ndarray | None:
    """One decoded phone ``HandFrame`` JSON object -> ``(21, 3)`` wrist-relative
    meters, or ``None`` if the frame can't be used as-is.

    ``AdaptiveAnalyticalRetargeter.retarget()`` takes its ``(21, 3)`` input
    as-is -- no confidence, no mask, no NaN handling -- so the only decision
    to make here is publish or drop, matching
    ``orca_teleop.ingress.server.IngressServer``'s own existing policy of
    dropping any frame it can't validate rather than trying to repair it.

    A frame is dropped if any of the 21 named joints is missing entirely,
    has ``confidence`` below ``min_confidence``, or has no ``position``
    (Vision had no depth reading for it, or for the wrist itself) -- never
    filled in or interpolated.
    """
    joints_by_name = {joint["name"]: joint for joint in frame.get("joints", []) if "name" in joint}

    points = np.empty((len(PHONE_JOINT_NAMES), 3), dtype=np.float32)
    for i, name in enumerate(PHONE_JOINT_NAMES):
        joint = joints_by_name.get(name)
        if joint is None:
            return None
        if joint.get("confidence", 0.0) < min_confidence:
            return None
        position = joint.get("position")
        if position is None:
            return None
        points[i] = (position["x"], position["y"], position["z"])

    return points
