"""ZMQ transport tests covering every registered RPC method.

The in-process transport is exercised in ``test_rpc_methods.py``; this file
proves the **wire format** — every method round-trips through a real ZeroMQ
REQ/REP socket, the response envelope carries ``ok``, and the result is
JSON-serialisable after transit.

Simulate and scan are tested with mocks (the real kernels take seconds and
the ZMQ server processes serially). Read-only methods are tested end-to-end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zmq

from quest.core import DecaySimulationResult
from quest.rpc.dispatcher import ServiceDispatcher
from quest.rpc.services import register_services
from quest.rpc.zmq_server import ZMQServer

REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = REPO_ROOT / "tests" / "148l.pdb"


@pytest.fixture(scope="module")
def server_port() -> int:
    """Pick an uncommon port to avoid clashing with a dev server."""
    return 5571


@pytest.fixture(scope="module")
def zmq_server(server_port) -> ZMQServer:
    dispatcher = ServiceDispatcher()
    register_services(dispatcher)
    server = ZMQServer(dispatcher, host="127.0.0.1", port=server_port)
    server.start(in_background=True)
    time.sleep(0.2)
    yield server
    server.stop()


class _ZMQClient:
    """Thin wrapper for a REQ socket that sends one request per call."""

    def __init__(self, port: int) -> None:
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://127.0.0.1:{port}")
        self.sock.setsockopt(zmq.RCVTIMEO, 10_000)

    def call(self, method: str, params: dict | None = None, req_id: int = 1) -> dict:
        request = json.dumps(
            {"method": method, "params": params or {}, "id": req_id}
        )
        self.sock.send_string(request)
        return json.loads(self.sock.recv_string())

    def close(self) -> None:
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture
def cli(zmq_server, server_port) -> _ZMQClient:
    client = _ZMQClient(server_port)
    yield client
    client.close()


class TestWireProtocol:
    def test_id_is_echoed(self, cli) -> None:
        r = cli.call("quest.template", {"with_fret": False}, req_id=99)
        assert r["id"] == 99

    def test_unknown_method(self, cli) -> None:
        r = cli.call("quest.nope")
        assert r["ok"] is False
        assert r["error_code"] == "unknown_method"

    def test_missing_method_field(self, cli) -> None:
        raw = json.dumps({"params": {}, "id": 1})
        cli.sock.send_string(raw)
        reply = json.loads(cli.sock.recv_string())
        assert reply["ok"] is False
        assert reply["error_code"] == "invalid_request"


class TestReadOnlyMethodsOverZMQ:
    """Every read-only method returns a well-formed envelope over the wire."""

    @pytest.mark.parametrize(
        "method,params,check",
        [
            (
                "quest.template",
                {"with_fret": False},
                lambda r: r["result"]["project"]["fret"]["enabled"] is False,
            ),
            (
                "quest.template",
                {"with_fret": True},
                lambda r: len(r["result"]["project"]["fret"]["dyes"]) == 2,
            ),
            (
                "quest.parameter_catalog",
                {},
                lambda r: r["result"]["version"] == 1,
            ),
            (
                "quest.parameter_catalog",
                {"locale": "de"},
                lambda r: r["result"]["version"] == 1,
            ),
            (
                "quest.locales",
                {},
                lambda r: "en" in r["result"]["available"],
            ),
            (
                "quest.locales",
                {"locale": "fr"},
                lambda r: r["result"]["locale"] == "fr",
            ),
            (
                "quest.dyes.list",
                {},
                lambda r: "Alexa488" in r["result"],
            ),
            (
                "quest.quenching_defaults",
                {},
                lambda r: len(r["result"]["residues"]) == 20,
            ),
            (
                "quest.jobs.list",
                {},
                lambda r: isinstance(r["result"]["jobs"], list),
            ),
            (
                "quest.contract.describe",
                {},
                lambda r: r["result"]["plugin_id"] == "quest",
            ),
            (
                "quest.runs.list",
                {},
                lambda r: isinstance(r["result"]["runs"], list),
            ),
        ],
    )
    def test_method(self, cli, method, params, check) -> None:
        r = cli.call(method, params)
        assert r["ok"] is True, f"{method} failed: {r.get('error')}"
        assert check(r), f"{method} check failed"


class TestValidateOverZMQ:
    def test_valid_project(self, cli) -> None:
        from quest import api

        project = api.template(with_fret=False)
        project["pdb"] = str(STRUCTURE)
        project["attachment"] = {"chain": "E", "residue": 96, "atom": "CB"}
        project["fret"]["dyes"][0]["attachment"] = dict(project["attachment"])
        r = cli.call("quest.validate", {"project": project})
        assert r["ok"] is True
        assert r["result"]["valid"] is True

    def test_invalid_project(self, cli) -> None:
        r = cli.call("quest.validate", {"project": {"tau0": 1.0}})
        assert r["ok"] is False
        assert r["error_code"] == "invalid_project"


class TestStructureMetadataOverZMQ:
    def test_valid_structure(self, cli) -> None:
        r = cli.call(
            "quest.structure.metadata", {"pdb_path": str(STRUCTURE)}
        )
        assert r["ok"] is True
        assert r["result"]["atom_count"] > 0

    def test_nonexistent(self, cli) -> None:
        r = cli.call(
            "quest.structure.metadata",
            {"pdb_path": "/nonexistent/file.pdb"},
        )
        assert r["ok"] is False


class TestSimulateOverZMQ:
    def test_simulate_payload_is_json_after_wire(self, cli, monkeypatch) -> None:
        """The simulate handler returns a native object; the ZMQ transport
        must serialise it so the reply is valid JSON.

        We mock ``api.simulate_project`` and ``validate_project`` to keep the
        test fast while exercising the full serialisation path.
        """

        def fake_simulate(*a, **kw):
            return DecaySimulationResult(
                time=np.array([0.0, 1.0, 2.0]),
                donor_counts=np.array([10.0, 5.0, 2.0]),
                fret_counts=None,
                quantum_yield_donor=0.5,
                quantum_yield_fret=None,
                collisions_fraction=0.1,
                fret_efficiency=None,
                project=a[0] if a else kw.get("project", {}),
            )

        import quest.project

        monkeypatch.setattr(quest.project, "validate_project", lambda *a, **kw: None)
        monkeypatch.setattr(
            "quest.api.simulate_project", fake_simulate
        )
        r = cli.call(
            "quest.simulate",
            {"project": {"_mock": True}},
        )
        assert r["ok"] is True
        assert r["result"]["donor_counts"] == [10.0, 5.0, 2.0]
        assert r["result"]["time_ns"] == [0.0, 1.0, 2.0]

    def test_malformed_project(self, cli) -> None:
        r = cli.call("quest.simulate", {"project": {"tau0": 1.0}})
        assert r["ok"] is False
        assert r["error_code"] == "invalid_project"


class TestJobsCancelOverZMQ:
    def test_cancel_unknown_job(self, cli) -> None:
        r = cli.call("quest.jobs.cancel", {"job_id": "no-such-job"})
        assert r["ok"] is True
        assert r["result"]["cancel_requested"] is False


class TestScanOverZMQ:
    """quest.scan round-trips through ZMQ with a mocked kernel."""

    def test_scan_returns_records_and_job_id(self, cli, monkeypatch) -> None:
        def fake_scan_positions(project, positions, **kw):
            for name in positions:
                yield {
                    "name": name,
                    "status": "completed",
                    "quantum_yield": 0.5,
                    "av_volume": 100.0,
                    "quenchers": [],
                    "error": None,
                }

        import quest.project
        monkeypatch.setattr(quest.project, "validate_project", lambda *a, **kw: None)
        monkeypatch.setattr("quest.scan.scan_positions", fake_scan_positions)

        r = cli.call(
            "quest.scan",
            {
                "project": {"_mock": True},
                "positions": {
                    "E96": {
                        "chain_identifier": "E",
                        "residue_seq_number": 96,
                        "atom_name": "CB",
                        "residue_name": "ALA",
                    }
                },
            },
        )
        assert r["ok"] is True
        assert r["result"]["job_id"] is not None
        assert "E96" in r["result"]["results"]
        assert r["result"]["results"]["E96"]["status"] == "completed"

    def test_scan_result_is_json_serialisable(self, cli, monkeypatch) -> None:
        def fake_scan_positions(project, positions, **kw):
            for name in positions:
                yield {"name": name, "status": "completed",
                       "quantum_yield": 0.5, "av_volume": 100.0,
                       "quenchers": [], "error": None}

        import quest.project
        monkeypatch.setattr(quest.project, "validate_project", lambda *a, **kw: None)
        monkeypatch.setattr("quest.scan.scan_positions", fake_scan_positions)

        r = cli.call("quest.scan", {"project": {"_mock": True}, "positions": {}})
        assert r["ok"] is True
        assert r["result"]["results"] == {}
        json.dumps(r["result"])


class TestDyesSaveOverZMQ:
    def test_nameless_dye_fails(self, cli) -> None:
        r = cli.call("quest.dyes.save", {"dye": {"tau0": 4.0}})
        assert r["ok"] is False
        assert "name" in r["error"]

    def test_valid_dye_is_saved(self, cli, monkeypatch) -> None:
        from quest import settings

        saved: dict = {}
        original = settings.dye_repository

        def fake_repository():
            dyes = original()
            if saved:
                dyes[saved["name"]] = saved
            return dyes

        monkeypatch.setattr(
            settings, "save_dye",
            lambda dye: saved.update(dye) or dye if "name" in dye else (_ for _ in ()).throw(ValueError("dye needs a name")),
        )
        monkeypatch.setattr(settings, "dye_repository", fake_repository)

        r = cli.call("quest.dyes.save", {"dye": {"name": "ZMQTestFluor", "tau0": 3.5}})
        assert r["ok"] is True


class TestRunsGetOverZMQ:
    def test_missing_run_fails(self, cli) -> None:
        r = cli.call("quest.runs.get", {"job_id": "nonexistent-uuid"})
        assert r["ok"] is False


class TestJobsGetOverZMQ:
    def test_nonexistent_job_fails(self, cli) -> None:
        r = cli.call("quest.jobs.get", {"job_id": "no-such-job"})
        assert r["ok"] is False
        assert r["error_code"] == "operation_failed"


class TestProcessMessageDirectly:
    """Exercise ``ZMQServer.process_message`` without a socket."""

    def test_dispatches_a_method(self) -> None:
        dispatcher = ServiceDispatcher()
        register_services(dispatcher)
        server = ZMQServer(dispatcher)
        raw = json.dumps(
            {"method": "quest.template", "params": {"with_fret": False}, "id": 7}
        )
        reply = json.loads(server.process_message(raw))
        assert reply["ok"] is True
        assert reply["id"] == 7

    def test_garbage_payload(self) -> None:
        dispatcher = ServiceDispatcher()
        register_services(dispatcher)
        server = ZMQServer(dispatcher)
        reply = json.loads(server.process_message("not json"))
        assert reply["ok"] is False
        # A malformed envelope is the caller's fault, not a failed operation.
        assert reply["error_code"] == "invalid_request"
        assert "result" not in reply

    def test_bytes_input(self) -> None:
        dispatcher = ServiceDispatcher()
        register_services(dispatcher)
        server = ZMQServer(dispatcher)
        raw = json.dumps({"method": "quest.template", "params": {}, "id": 1})
        reply = json.loads(server.process_message(raw.encode("utf-8")))
        assert reply["ok"] is True
