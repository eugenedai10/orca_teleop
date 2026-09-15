"""Bridge HandTeleopCapture's phone stream to the orca_teleop gRPC ingress.

Run this on the same machine as the teleop pipeline (see
``pipeline.run_phone_local``, or ``--source phone`` on
``scripts/teleop_sim.py``), pointed at the phone's LAN IP (shown in the
HandTeleopCapture app's UI). It's a pure publisher, like
``MediaPipePublisher``/``MetaQuestPublisher``: it knows nothing about the
robot, it just turns the phone's NDJSON stream into ``HandFrame`` protos.

Usage::

    python -m orca_teleop.ingress.phone.publisher --phone-host 192.168.1.50
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import grpc

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.phone.client import DEFAULT_PHONE_PORT, PhoneStreamClient
from orca_teleop.ingress.phone.landmarks import DEFAULT_MIN_CONFIDENCE, landmarks_from_phone_frame

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "localhost:50051"
RECONNECT_WAIT_S = 2.0
_LOG_EVERY_S = 5.0


class PhonePublisher:
    """HandTeleopCapture phone-stream ingress -> gRPC publisher."""

    def __init__(
        self,
        server_address: str = DEFAULT_SERVER,
        *,
        phone_host: str,
        phone_port: int = DEFAULT_PHONE_PORT,
        handedness: str = "right",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        client: PhoneStreamClient | None = None,
    ) -> None:
        if handedness not in ("left", "right"):
            raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")
        self._server_address = server_address
        self._handedness = handedness
        self._min_confidence = float(min_confidence)
        self._client = client or PhoneStreamClient(phone_host, phone_port)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _frame_generator(self):
        received = 0
        published = 0
        dropped = 0
        next_log = time.monotonic() + _LOG_EVERY_S

        while not self._stop.is_set():
            try:
                self._client.connect()
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Waiting for phone at %s -- open HandTeleopCapture on the "
                    "same network (%s)",
                    self._client.address,
                    exc,
                )
                time.sleep(RECONNECT_WAIT_S)
                continue

            try:
                for frame in self._client.frames():
                    if self._stop.is_set():
                        return

                    received += 1
                    keypoints = landmarks_from_phone_frame(frame, self._min_confidence)
                    if keypoints is None:
                        dropped += 1
                    else:
                        published += 1
                        yield hand_stream_pb2.HandFrame(
                            keypoints=keypoints.ravel().tolist(),
                            handedness=self._handedness,
                            timestamp_ns=time.time_ns(),
                            wrist_angle_degrees=0.0,
                        )

                    now = time.monotonic()
                    if now >= next_log:
                        logger.info(
                            "Phone publisher: %d received, %d published, %d dropped",
                            received,
                            published,
                            dropped,
                        )
                        next_log = now + _LOG_EVERY_S
            except (ConnectionError, OSError) as exc:
                logger.warning("Phone connection lost (%s); reconnecting.", exc)
                time.sleep(RECONNECT_WAIT_S)

    def run(self) -> None:
        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        logger.info(
            "Connecting phone publisher to %s (hand=%s, min_confidence=%.2f)",
            self._server_address,
            self._handedness,
            self._min_confidence,
        )
        try:
            summary = stub.StreamHandFrames(self._frame_generator(), wait_for_ready=True)
            logger.info(
                "Phone stream closed: ingress received %d frames.", summary.frames_received
            )
        except KeyboardInterrupt:
            logger.info("Phone publisher interrupted.")
        except grpc.RpcError as exc:
            if not self._stop.is_set():
                logger.error("Phone publisher gRPC error: %s", exc)
        finally:
            self._stop.set()
            channel.close()
            self._client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC ingress host:port")
    parser.add_argument(
        "--phone-host", required=True, help="Phone's LAN IP (shown in HandTeleopCapture)"
    )
    parser.add_argument("--phone-port", type=int, default=DEFAULT_PHONE_PORT)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Per-joint confidence gate below which a frame is dropped (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    PhonePublisher(
        server_address=args.server,
        phone_host=args.phone_host,
        phone_port=args.phone_port,
        handedness=args.hand,
        min_confidence=args.min_confidence,
    ).run()


if __name__ == "__main__":
    main()
