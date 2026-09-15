"""TCP client for HandTeleopCapture's NDJSON phone stream.

Connects to the phone app's own server (raw TCP, port 9000 by default, one
JSON object per line -- see
`teleop/HandTeleopCapture/Data/StreamingServer.swift`). This is a fresh,
minimal implementation: `orca_teleop` does not import anything from the
`teleop` submodule at runtime, so this ingress has no dependency on it beyond
the wire format documented in `HandFrame.swift`.
"""

from __future__ import annotations

import json
import logging
import socket
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PHONE_PORT = 9000
_RECV_BUFFER_SIZE = 65536


class PhoneStreamClient:
    """Reads newline-delimited JSON ``HandFrame``s from a HandTeleopCapture session.

    The stream has no framing beyond the newline, so a single ``recv()`` can
    split one JSON object across two reads, or return several objects in one
    read -- this buffers across calls rather than assuming one ``recv()`` is
    one line.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PHONE_PORT,
        connect_timeout: float = 5.0,
    ) -> None:
        self._host = host
        self._port = port
        self._connect_timeout = connect_timeout
        self._sock: socket.socket | None = None
        self._buffer = b""

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"

    def connect(self) -> dict[str, Any]:
        """Open the connection and return the session header (line 1)."""
        self.close()
        sock = socket.create_connection((self._host, self._port), timeout=self._connect_timeout)
        self._sock = sock
        self._buffer = b""
        header = json.loads(self._read_line())
        logger.info(
            "Connected to phone %s (device=%s, video=%sx%s)",
            self.address,
            header.get("deviceModel"),
            header.get("videoWidth"),
            header.get("videoHeight"),
        )
        return header

    def frames(self) -> Iterator[dict[str, Any]]:
        """Yield decoded ``HandFrame`` dicts until the connection drops.

        Raises ``ConnectionError`` (peer closed) or ``OSError`` (network
        error) the same way the underlying socket does -- callers decide
        whether/how to reconnect.
        """
        while True:
            line = self._read_line()
            try:
                yield json.loads(line)
            except ValueError:
                logger.warning("Dropping malformed line from phone stream")
                continue

    def _read_line(self) -> bytes:
        assert self._sock is not None, "connect() must be called first"
        while b"\n" not in self._buffer:
            chunk = self._sock.recv(_RECV_BUFFER_SIZE)
            if chunk == b"":
                raise ConnectionError(f"phone at {self.address} closed the connection")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
