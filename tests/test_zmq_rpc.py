from __future__ import annotations

import json
import time
import zmq

from quest.rpc.dispatcher import ServiceDispatcher
from quest.rpc.services import register_services
from quest.rpc.zmq_server import ZMQServer


def test_zmq_server_basic_rpc():
    dispatcher = ServiceDispatcher()
    register_services(dispatcher)

    port = 5599
    server = ZMQServer(dispatcher, host="127.0.0.1", port=port)
    server.start(in_background=True)
    time.sleep(0.1)  # Allow socket bind to complete

    try:
        ctx = zmq.Context()
        client = ctx.socket(zmq.REQ)
        client.connect(f"tcp://127.0.0.1:{port}")
        client.setsockopt(zmq.RCVTIMEO, 3000)

        # Test method: quest.template
        req = json.dumps({"method": "quest.template", "params": {"with_fret": True}, "id": 42})
        client.send_string(req)
        reply = json.loads(client.recv_string())

        assert reply.get("ok") is True
        assert reply.get("id") == 42
        assert "project" in reply.get("result", {})

        # Test unknown method
        req2 = json.dumps({"method": "quest.unknown_foo", "params": {}})
        client.send_string(req2)
        reply2 = json.loads(client.recv_string())

        assert reply2.get("ok") is False
        assert reply2.get("error_code") == "unknown_method"

        client.close(linger=0)
        ctx.term()
    finally:
        server.stop()
