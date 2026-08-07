"""ZeroMQ RPC Server for QuEst.

Exposes QuEst's ServiceDispatcher over a ZeroMQ REP socket, allowing
out-of-process and cross-language remote procedure calls.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

import zmq

from quest.rpc.dispatcher import ServiceDispatcher, fail

_log = logging.getLogger(__name__)


class ZMQServer:
    """ZeroMQ RPC Server wrapping a ServiceDispatcher."""

    def __init__(
        self,
        dispatcher: ServiceDispatcher,
        host: str = "127.0.0.1",
        port: int = 5555,
    ) -> None:
        self.dispatcher = dispatcher
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def process_message(self, raw_message: str | bytes) -> str:

        """Parse raw request, execute RPC method via dispatcher, return JSON string."""
        req_id = None
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            data = json.loads(raw_message)
            req_id = data.get("id")
            method = data.get("method")
            params = data.get("params", {})

            if not method:
                response = fail("Missing 'method' in RPC request", "invalid_request")
            else:
                response = self.dispatcher.dispatch(method, params)
        except Exception as exc:
            _log.exception("Failed to process ZMQ message")
            response = fail(f"Invalid ZMQ request payload: {exc}", "invalid_payload")

        if req_id is not None:
            response["id"] = req_id

        return json.dumps(response)

    def serve_one(self, timeout_ms: int = 1000) -> bool:
        """Poll and serve a single incoming request if available within timeout_ms."""
        if self.socket is None:
            return False

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        events = dict(poller.poll(timeout_ms))

        if self.socket in events and events[self.socket] == zmq.POLLIN:
            msg = self.socket.recv_string()
            response_json = self.process_message(msg)
            self.socket.send_string(response_json)
            return True
        return False

    def start(self, in_background: bool = True) -> None:
        """Bind and start serving RPC requests."""
        if self._running:
            return

        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.endpoint)
        self._running = True
        _log.info(f"ZMQ RPC Server bound to {self.endpoint}")

        if in_background:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        else:
            self._run_loop()

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.serve_one(timeout_ms=500)
            except zmq.ZMQError as exc:
                if not self._running:
                    break
    def stop(self) -> None:
        """Stop serving and close socket."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self.socket = None
        _log.info("ZMQ RPC Server stopped")

