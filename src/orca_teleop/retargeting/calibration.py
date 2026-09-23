"""Explicit retargeting calibration: a dedicated, operator-driven process that
locks in ``AdaptiveAnalyticalRetargeter``'s ``mano_scale`` before a teleop
session starts, instead of the implicit, silent auto-calibration that
otherwise runs on whatever pose the operator happens to hold during the
first ~1s of live frames (see ``worklog.md``'s 2026-09-16 entry for the bug
this replaces).

Structured like orca_core's maintenance routines (``calibration_routine.py``,
``tensioning.py``): a pure routine driven by ``progress_callback``/
``prompt_callback``, with zero I/O of its own, so a CLI script and
(eventually) a GUI can drive identical logic. The routine reuses
``AdaptiveAnalyticalRetargeter``'s own ``retarget()``/``reset()`` -- the scale
math itself is untouched.

This is a dedicated, one-time process: calibrate once via
``scripts/calibrate_retargeting.py``, persist the result keyed by operator,
then driving sessions (``teleop_real.py`` etc.) load it instead of
recalibrating. There is no live/mid-session recalibration trigger.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml

from orca_teleop.retargeting.constants import CALIBRATION_FRAMES
from orca_teleop.retargeting.retargeter import TargetPose

if TYPE_CHECKING:
    from orca_teleop.ingress.server import HandLandmarks
    from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]
PromptCallback = Callable[[dict], None]
ShouldStop = Callable[[], bool]

DEFAULT_COUNTDOWN_S = 3.0
DEFAULT_FRAME_TIMEOUT_S = 2.0

_PROFILE_DIR_ENV = "ORCA_TELEOP_CALIBRATION_DIR"
_PROFILE_DIR_DEFAULT = Path.home() / ".orca_teleop" / "retarget_calibration"


class RetargetCalibrationError(RuntimeError):
    """Raised when the calibration routine cannot proceed without a human action."""


def _emit(progress_callback: ProgressCallback | None, event: str, **payload) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback({"event": event, **payload})
    except Exception:
        logger.exception("retarget calibration progress callback failed")


def _prompt(prompt_callback: PromptCallback | None, action: str, **payload) -> None:
    if prompt_callback is None:
        raise RetargetCalibrationError(
            f"retarget calibration needs to ask the operator to {action!r}, "
            "but no prompt_callback was supplied"
        )
    prompt_callback({"action": action, **payload})


def run_retarget_calibration(
    retargeter: "AdaptiveAnalyticalRetargeter",
    landmarks_q: "queue.Queue[HandLandmarks]",
    *,
    countdown_s: float = DEFAULT_COUNTDOWN_S,
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    progress_callback: ProgressCallback | None = None,
    prompt_callback: PromptCallback | None = None,
    should_stop: ShouldStop | None = None,
) -> float | None:
    """Explicitly (re)calibrate ``retargeter``'s ``mano_scale`` against live frames.

    Blocks the calling thread: prompts the operator to hold the robot's
    neutral pose (via ``prompt_callback``), counts down, then pulls frames off
    ``landmarks_q`` and feeds them through the retargeter's own ``retarget()``
    until its internal calibration window completes. Returns the resulting
    ``mano_scale``, or ``None`` if aborted, or the frame stream stalls/ends.

    Meant to run standalone, before any worker thread starts consuming
    ``landmarks_q`` -- see ``scripts/calibrate_retargeting.py``. Calls
    ``retargeter.reset()`` first, so any partial/implicit calibration state
    is discarded before sampling begins.
    """
    if should_stop is None:
        should_stop = lambda: False  # noqa: E731

    retargeter.reset()
    _emit(progress_callback, "calibration_started")

    _prompt(
        prompt_callback,
        "confirm_neutral_pose",
        message="Hold the robot's neutral/rest hand pose and confirm when ready.",
    )

    countdown_ticks = max(0, int(round(countdown_s)))
    for remaining in range(countdown_ticks, 0, -1):
        if should_stop():
            _emit(progress_callback, "calibration_aborted")
            return None
        _emit(progress_callback, "countdown", seconds_remaining=remaining)
        time.sleep(1.0)

    frames_seen = 0
    while not retargeter._calibration_done:
        if should_stop():
            _emit(progress_callback, "calibration_aborted", frames_seen=frames_seen)
            return None
        try:
            landmarks = landmarks_q.get(timeout=frame_timeout_s)
        except queue.Empty:
            _emit(progress_callback, "calibration_timed_out", frames_seen=frames_seen)
            return None
        if not hasattr(landmarks, "keypoints"):
            # Shutdown sentinel or other unexpected item on the queue.
            _emit(progress_callback, "calibration_aborted", frames_seen=frames_seen)
            return None

        target_pose = TargetPose(
            joint_positions=landmarks.keypoints,
            source="mediapipe",
            wrist_angle_degrees=getattr(landmarks, "wrist_angle_degrees", 0.0),
        )
        try:
            retargeter.retarget(target_pose)
        except (AssertionError, ValueError):
            logger.debug("Skipping degenerate landmark frame during calibration.")
            continue

        frames_seen += 1
        _emit(progress_callback, "sample_recorded", index=frames_seen, total=CALIBRATION_FRAMES)

    _emit(
        progress_callback,
        "calibration_done",
        mano_scale=retargeter._mano_scale,
        frames_seen=frames_seen,
    )
    return retargeter._mano_scale


@dataclass(frozen=True)
class CalibrationProfile:
    """A persisted, per-operator retargeting calibration result."""

    operator: str
    mano_scale: float
    model_path: str
    urdf_path: str
    calibrated_at: str  # ISO 8601
    frames_used: int
    finger_ratios: dict[str, float] | None = None  # diagnostic only, not reapplied on load


def profile_dir() -> Path:
    return Path(os.environ.get(_PROFILE_DIR_ENV, str(_PROFILE_DIR_DEFAULT)))


def default_profile_path(operator: str) -> Path:
    return profile_dir() / f"{operator}.yaml"


def resolve_profile_path(operator_or_path: str) -> Path:
    """Accept either a bare operator name or an explicit ``.yaml``/``.yml`` path."""
    candidate = Path(operator_or_path)
    if candidate.suffix in (".yaml", ".yml") or candidate.is_absolute():
        return candidate
    return default_profile_path(operator_or_path)


def save_profile(path: Path, profile: CalibrationProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(asdict(profile), f, sort_keys=False)
    tmp_path.replace(path)


def load_profile(path: Path) -> CalibrationProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"No retargeting calibration profile at {path}. Run "
            "`scripts/calibrate_retargeting.py --operator <name> --model-path ...` first."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return CalibrationProfile(**data)


def apply_profile(retargeter: "AdaptiveAnalyticalRetargeter", profile: CalibrationProfile) -> None:
    retargeter.load_mano_scale(profile.mano_scale)
    logger.info(
        "Applied retargeting calibration profile for operator=%r: mano_scale=%.4f",
        profile.operator,
        profile.mano_scale,
    )


def default_progress_callback(event: dict) -> None:
    """Print-based progress renderer -- the CLI's default when no callback is given."""
    name = event.get("event")
    if name == "calibration_started":
        print("Retargeting calibration starting...")
    elif name == "countdown":
        print(f"  starting in {event['seconds_remaining']}...")
    elif name == "sample_recorded":
        print(f"  sample {event['index']}/{event['total']}", end="\r")
    elif name == "calibration_done":
        print(f"\nRetargeting calibrated: mano_scale={event['mano_scale']:.4f}")
    elif name == "calibration_aborted":
        print("\nRetargeting calibration aborted.")
    elif name == "calibration_timed_out":
        print(f"\nRetargeting calibration timed out after {event['frames_seen']} frame(s).")


def default_prompt_callback(prompt: dict) -> None:
    """``input()``-based confirmation, TTY-gated like ``teleop_real.py``'s ``--yes`` gate."""
    if prompt["action"] != "confirm_neutral_pose":
        return
    print(prompt.get("message", "Hold the robot's neutral pose."))
    if not sys.stdin.isatty():
        raise RetargetCalibrationError(
            "Refusing to run retargeting calibration without confirmation in a "
            "non-interactive shell."
        )
    input("Press ENTER when ready... ")
