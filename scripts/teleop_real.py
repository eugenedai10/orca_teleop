"""Teleoperate the real, physical ORCA hand via the same ingress + retargeter
stack used for the simulator. The only difference from ``teleop_sim.py`` is
the sink: instead of stepping an ``orca_sim`` env, ``OrcaHandSink`` drives an
``orca_core.OrcaHand`` directly.

This moves real hardware: as soon as you confirm the pre-flight prompt below,
the hand enables torque and moves to its neutral pose.

Examples:
    # Robot machine: start the pipeline against the real hand and wait for a
    # remote publisher to connect over gRPC.
    python scripts/teleop_real.py --model_path <calibrated model dir> --env right

    # Laptop: one-command teleop with a local MediaPipe webcam publisher.
    python scripts/teleop_real.py --model_path <calibrated model dir> --local --show-video
"""

import argparse
import logging
import sys

from orca_teleop.ingress.phone.client import DEFAULT_PHONE_PORT
from orca_teleop.ingress.phone.landmarks import DEFAULT_MIN_CONFIDENCE
from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import OrcaHandSink, run, run_local, run_phone_local


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orca Hand teleoperation pipeline against the real hardware."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Calibrated OrcaHand model directory (e.g. orca_core/orca_core/models/v2/orcahand-right)",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help=(
            "Retargeting calibration profile to load (see scripts/calibrate_retargeting.py). "
            "Real hardware always requires a profile -- there is no implicit fallback."
        ),
    )
    parser.add_argument("--urdf_path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also launch a local publisher (--source) for one-command teleop",
    )
    parser.add_argument(
        "--source",
        default="mediapipe",
        choices=["mediapipe", "phone"],
        help="Local publisher to launch with --local (default: mediapipe)",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Hand to track in the local publisher (default: right)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="MediaPipe confidence, --source mediapipe only (default: 0.7)",
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument(
        "--phone-host",
        default=None,
        help="Phone's LAN IP running HandTeleopCapture, --source phone only",
    )
    parser.add_argument(
        "--phone-port",
        type=int,
        default=DEFAULT_PHONE_PORT,
        help=f"Phone stream TCP port, --source phone only (default: {DEFAULT_PHONE_PORT})",
    )
    parser.add_argument(
        "--phone-min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=(
            "Per-joint confidence gate for the phone source, below which a "
            f"frame is dropped (default: {DEFAULT_MIN_CONFIDENCE})"
        ),
    )
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retarget-config",
        default=None,
        help="YAML config for --retargeter adaptive_analytical",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation before enabling torque (for scripted use)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.yes:
        print(f"About to connect to the REAL ORCA hand using model_path={args.model_path!r}.")
        print("Torque will enable and the hand will move to its neutral pose immediately.")
        if not sys.stdin.isatty():
            parser.error(
                "Refusing to drive real hardware without confirmation in a non-interactive "
                "shell. Re-run with --yes to skip the prompt."
            )
        reply = input("Continue? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return

    sink = OrcaHandSink(model_path=args.model_path)

    if args.local and args.source == "phone":
        if not args.phone_host:
            parser.error("--phone-host is required with --local --source phone")
        run_phone_local(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            handedness=args.hand,
            phone_host=args.phone_host,
            phone_port=args.phone_port,
            min_confidence=args.phone_min_confidence,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
            retarget_calibration_profile=args.operator,
        )
    elif args.local:
        run_local(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            handedness=args.hand,
            confidence=args.confidence,
            show_video=args.show_video,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
            retarget_calibration_profile=args.operator,
        )
    else:
        run(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
            retarget_calibration_profile=args.operator,
        )


if __name__ == "__main__":
    main()
