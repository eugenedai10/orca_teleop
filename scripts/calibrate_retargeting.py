"""Dedicated, standalone retargeting calibration.

Measures this operator's ``mano_scale`` once, against live landmark frames,
and persists it -- so driving sessions (``teleop_real.py`` etc., via
``--operator``) load a known-good scale instead of the retargeter's implicit,
silent auto-calibration on whatever pose happens to be held during the first
~1s of live frames (see ``worklog.md``'s 2026-09-16 entry for the bug this
replaces).

This is a one-time process per operator, not something re-run mid-session:
run it once, then start ``teleop_real.py``/``teleop_sim.py``/``teleop_urdf.py``
with the same ``--operator`` name.

Two ways to run, same shape as ``retargeter_diagnostics.py``/``teleop_real.py``:

    # Webcam / MediaPipe source (spawns its own publisher):
    python scripts/calibrate_retargeting.py --operator eugene \\
        --model-path /path/to/orcahand-right/config.yaml \\
        --urdf-path candidate_orcahand_right_teleop.urdf \\
        --local --show-video

    # Phone source:
    python scripts/calibrate_retargeting.py --operator eugene \\
        --model-path ... --urdf-path ... \\
        --local --source phone --phone-host <phone's LAN IP>

    # Bring your own publisher (no --local):
    python scripts/calibrate_retargeting.py --operator eugene --model-path ... --urdf-path ...
    python -m orca_teleop.ingress.mediapipe.publisher --server localhost:50051

Saves a ``CalibrationProfile`` YAML to
``~/.orca_teleop/retarget_calibration/<operator>.yaml`` by default -- override
the directory with ``$ORCA_TELEOP_CALIBRATION_DIR``, or the exact path with
``--profile-path``.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import queue
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from orca_teleop.constants import QUEUES_MAXSIZE
from orca_teleop.ingress.phone.client import DEFAULT_PHONE_PORT
from orca_teleop.ingress.phone.landmarks import DEFAULT_MIN_CONFIDENCE
from orca_teleop.ingress.server import DEFAULT_PORT, IngressServer
from orca_teleop.pipeline import _mediapipe_publisher, _phone_publisher
from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.calibration import (
    CalibrationProfile,
    default_prompt_callback,
    default_progress_callback,
    default_profile_path,
    run_retarget_calibration,
    save_profile,
)
from orca_teleop.retargeting.constants import CALIBRATION_FRAMES

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator", required=True, help="Name to save this calibration profile under"
    )
    parser.add_argument(
        "--model-path", required=True, help="Calibrated OrcaHand model directory or config.yaml"
    )
    parser.add_argument("--urdf-path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--retarget-config", default=None, help="YAML config for the adaptive_analytical retargeter"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC ingress port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--local", action="store_true", help="Also launch a local publisher (--source)"
    )
    parser.add_argument(
        "--source",
        default="mediapipe",
        choices=["mediapipe", "phone"],
        help="Local publisher to launch with --local (default: mediapipe)",
    )
    parser.add_argument(
        "--hand", default="right", choices=["left", "right"], help="Hand tracked (default: right)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument(
        "--phone-host",
        default=None,
        help="Phone's LAN IP running HandTeleopCapture, --source phone only",
    )
    parser.add_argument("--phone-port", type=int, default=DEFAULT_PHONE_PORT)
    parser.add_argument("--phone-min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        "--countdown", type=float, default=3.0, help="Seconds between confirm and sampling"
    )
    parser.add_argument(
        "--profile-path",
        default=None,
        help="Explicit save path; default is ~/.orca_teleop/retarget_calibration/<operator>.yaml",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.local and args.source == "phone" and not args.phone_host:
        parser.error("--phone-host is required with --local --source phone")

    landmarks_q: "queue.Queue" = queue.Queue(maxsize=QUEUES_MAXSIZE)
    stop_event = threading.Event()
    server = IngressServer(landmarks_q, stop_event, port=args.port)
    server.start()

    publisher_proc: multiprocessing.Process | None = None
    if args.local:
        if args.source == "phone":
            publisher_proc = multiprocessing.Process(
                target=_phone_publisher,
                args=(args.port, args.hand, args.phone_host, args.phone_port, args.phone_min_confidence),
                name="phone-publisher",
                daemon=True,
            )
        else:
            publisher_proc = multiprocessing.Process(
                target=_mediapipe_publisher,
                args=(args.port, args.hand, args.confidence, args.show_video),
                name="mediapipe-publisher",
                daemon=True,
            )
        publisher_proc.start()
        logger.info("Local %s publisher started (pid=%d)", args.source, publisher_proc.pid)

    try:
        retargeter = AdaptiveAnalyticalRetargeter.from_paths(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            config_path=args.retarget_config,
        )
    except Exception:
        logger.exception("Retargeter init failed.")
        stop_event.set()
        server.stop()
        if publisher_proc is not None and publisher_proc.is_alive():
            publisher_proc.terminate()
        sys.exit(1)

    try:
        mano_scale = run_retarget_calibration(
            retargeter,
            landmarks_q,
            countdown_s=args.countdown,
            progress_callback=default_progress_callback,
            prompt_callback=default_prompt_callback,
            should_stop=stop_event.is_set,
        )
    finally:
        stop_event.set()
        server.stop()
        if publisher_proc is not None and publisher_proc.is_alive():
            publisher_proc.terminate()
            publisher_proc.join(timeout=3.0)

    if mano_scale is None:
        print("Calibration did not complete; nothing saved.")
        sys.exit(1)

    profile = CalibrationProfile(
        operator=args.operator,
        mano_scale=mano_scale,
        model_path=args.model_path,
        urdf_path=retargeter.urdf_path,
        calibrated_at=datetime.now(timezone.utc).isoformat(),
        frames_used=CALIBRATION_FRAMES,
        finger_ratios=retargeter.calibration_ratios,
    )
    profile_path = Path(args.profile_path) if args.profile_path else default_profile_path(args.operator)
    save_profile(profile_path, profile)
    print(f"Saved retargeting calibration for operator={args.operator!r} to {profile_path}")


if __name__ == "__main__":
    main()
