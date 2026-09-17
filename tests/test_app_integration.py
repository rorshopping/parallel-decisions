"""Public integration contracts, using loopback HTTP and no model/runtime loads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import importlib.util
import json
import os
from pathlib import Path
import threading

import pytest

from parallel_decisions import Config, ConcurrencyError, Decider, DecisionResult, FieldValue, Schema
from parallel_decisions import cli

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"needs_human": {"type": "boolean", "description": "Requires review"}}


def load_example(name):
    spec = importlib.util.spec_from_file_location(f"integration_{name}", ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(calibrated=False):
    return DecisionResult(
        {"needs_human": FieldValue("needs_human", True, 0.812345, [],
                                   raw_probability=0.95, calibrated=calibrated)},
        model="test/model", latency_ms=12.34, calibrated=calibrated,
    )


@pytest.fixture
def decider(monkeypatch):
    d = Decider(model_id="test/model", backend="torch", config=Config(), warmup=False)
    d._model = object()  # load() returns immediately; no Torch/MLX import
    monkeypatch.setattr(d, "_decide_torch_locked", lambda *args: result())
    return d


@pytest.fixture
def serve():
    return load_example("serve")


@contextmanager
def running_server(serve, decider):
    serve.DECIDER = decider
    server = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()

    def request(method="POST", path="/decide", payload=None, body=None, headers=None):
        if body is None and payload is not None:
            body = json.dumps(payload).encode()
        conn = http.client.HTTPConnection(*server.server_address, timeout=3)
        try:
            conn.request(method, path, body=body, headers=headers or {"Content-Type": "application/json"})
            response = conn.getresponse()
            raw = response.read()
            assert response.getheader("Content-Type") == "application/json"
            assert int(response.getheader("Content-Length")) == len(raw)
            return response.status, json.loads(raw)
        finally:
            conn.close()

    try:
        yield request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.mark.parametrize("schema", [FIELDS, {"fields": FIELDS},
                                    json.loads(Schema(FIELDS).to_json())])
def test_http_success_schema_forms(serve, decider, schema):
    with running_server(serve, decider) as request:
        code, payload = request(payload={"context": "Ticket: café", "schema": schema})
        assert code == 200
        assert payload == {"decisions": {"needs_human": {"value": True, "probability": 0.8123}},
                           "calibrated": False, "model": "test/model", "latency_ms": 12.3}
        assert request("GET", "/health")[1] == {
            "ok": True, "model": "test/model", "busy": False,
            "served": 1, "queued": 0, "rejected": 0,
        }


@pytest.mark.parametrize("payload", [None, [], {}, {"context": "x"},
    {"context": "x", "schema": {}}, {"context": "x", "schema": []},
    {"context": "x", "schema": {"fields": [{"type": "boolean"}]}},
    *({"context": context, "schema": FIELDS} for context in (None, 12, [], "", "  "))])
def test_http_bad_request_is_json_400(serve, decider, payload):
    with running_server(serve, decider) as request:
        code, data = request(body=json.dumps(payload).encode())
        assert code == 400
        assert data["error"].startswith("bad request:")
        assert serve.STATS["served"] == 0


@pytest.mark.parametrize("body,headers", [(b"{", {}), (b"\xff", {}),
    (b"", {"Content-Length": "-1"}), (b"", {"Content-Length": "1000001"}),
    (b"", {"Content-Length": "oops"})])
def test_http_malformed_body(serve, decider, body, headers):
    with running_server(serve, decider) as request:
        assert request(body=body, headers=headers)[0] == 400


def test_http_unknown_routes(serve, decider):
    with running_server(serve, decider) as request:
        for method, path in (("GET", "/decide"), ("POST", "/health"), ("GET", "/missing")):
            assert request(method, path) == (404, {"error": "use POST /decide"})


def test_http_not_ready(serve):
    with running_server(serve, None) as request:
        assert request("GET", "/health")[1]["ok"] is False
        assert request(payload={"context": "x", "schema": FIELDS}) == (
            503, {"error": "model is not ready", "retry": True})


def test_http_busy_timeout_and_recovery(serve, decider):
    decider.lock_timeout_s = 0.01
    with running_server(serve, decider) as request:
        with decider._lock:
            assert request("GET", "/health")[1]["busy"] is True
            code, data = request(payload={"context": "x", "schema": FIELDS})
            assert code == 503 and data["retry"] is True
            assert "another thread" in data["error"]
        assert request(payload={"context": "x", "schema": FIELDS})[0] == 200
        assert serve.STATS == {"served": 1, "rejected": 1, "queued": 1}


def test_http_queued_request_waits_then_succeeds(serve, decider, monkeypatch):
    decider.lock_timeout_s = 2
    queued = threading.Event()
    original = serve._bump

    def bump(key):
        original(key)
        if key == "queued":
            queued.set()

    monkeypatch.setattr(serve, "_bump", bump)
    with running_server(serve, decider) as request, ThreadPoolExecutor(max_workers=1) as pool:
        with decider._lock:
            future = pool.submit(request, payload={"context": "x", "schema": FIELDS})
            assert queued.wait(timeout=2)
            assert not future.done()
        assert future.result(timeout=3)[0] == 200
        assert serve.STATS == {"served": 1, "rejected": 0, "queued": 1}


def test_http_runtime_failure_releases_lock(serve, decider, monkeypatch):
    def fail(*args):
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(decider, "_decide_torch_locked", fail)
    with running_server(serve, decider) as request:
        assert request(payload={"context": "x", "schema": FIELDS}) == (
            500, {"error": "runtime failed"})
        assert not decider._lock.locked()
        monkeypatch.setattr(decider, "_decide_torch_locked", lambda *args: result(True))
        code, payload = request(payload={"context": "x", "schema": FIELDS})
        assert code == 200 and payload["calibrated"] is True
        assert payload["decisions"]["needs_human"]["raw_probability"] == 0.95


@pytest.fixture
def clean_config(tmp_path, monkeypatch):
    for key in os.environ:
        if key.startswith("PD_"):
            monkeypatch.delenv(key)
    path = tmp_path / "pd.toml"
    path.write_text('model = "file/model"\nbackend = "torch"\n'
                    'torch_device = "cpu"\ntorch_dtype = "float32"\ncuda_graph = false\n',
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("timeout,file_timeout,env_timeout,expected", [
    (None, None, None, 600), (None, 3.5, None, 3.5),
    (None, 3.5, "2.5", 2.5), ("0", 3.5, "2.5", 0), ("1.5", None, None, 1.5),
])
def test_server_config_plumbing(serve, clean_config, monkeypatch,
                                timeout, file_timeout, env_timeout, expected):
    created = []
    servers = []

    def factory(**kwargs):
        d = Decider(**kwargs)
        monkeypatch.setattr(d, "load", lambda: d)
        created.append(d)
        return d

    class Server:
        def __init__(self, address, handler):
            servers.append(address)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def serve_forever(self):
            pass

    monkeypatch.setattr(serve, "Decider", factory)
    monkeypatch.setattr(serve, "ThreadingHTTPServer", Server)
    monkeypatch.setenv("PD_MODEL", "env/model")
    if file_timeout is not None:
        with clean_config.open("a", encoding="utf-8") as fh:
            fh.write(f"lock_timeout_s = {file_timeout}\n")
    if env_timeout is not None:
        monkeypatch.setenv("PD_LOCK_TIMEOUT_S", env_timeout)
    args = ["--config", str(clean_config), "--model", "explicit/model"]
    if timeout is not None:
        args += ["--queue-timeout", timeout]
    serve.main(args)
    d = created[0]
    assert d.model_id == "explicit/model" and d.backend == "torch"
    assert (d.torch_device, d.torch_dtype, d.cuda_graph) == ("cpu", "float32", False)
    assert d.lock_timeout_s == expected
    assert servers == [("127.0.0.1", 8000)]


@pytest.mark.parametrize("verbose", [False, True])
def test_cli_json_stdout_and_config(clean_config, tmp_path, monkeypatch, capsys, verbose):
    schema_path = tmp_path / "schema.json"
    Schema(FIELDS).to_json(str(schema_path))
    created = []

    def factory(**kwargs):
        d = Decider(**kwargs)
        created.append(d)
        def decide(context, schema):
            assert context == "x" and list(schema) == ["needs_human"]
            d._log("loading model")
            return result(True)
        monkeypatch.setattr(d, "decide", decide)
        return d

    monkeypatch.setattr(cli, "Decider", factory)
    monkeypatch.setenv("PD_MODEL", "env/model")
    args = ["--config", str(clean_config), "--model", "explicit/model", "decide",
            "--schema", str(schema_path), "--context", "x", "--json"]
    if verbose:
        args.append("--verbose")
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"decisions": result(True).full_json(), "calibrated": True}
    assert ("loading model" in captured.err) is verbose
    assert created[0].model_id == "explicit/model"
    assert created[0].torch_device == "cpu" and created[0].cuda_graph is False


@pytest.mark.parametrize("context", ["", "  "])
def test_cli_empty_context_errors_without_loading(tmp_path, monkeypatch, capsys, context):
    path = tmp_path / "schema.json"
    Schema(FIELDS).to_json(str(path))
    monkeypatch.setattr(cli, "Decider", lambda **kwargs: pytest.fail("must validate before model load"))
    assert cli.main(["decide", "--schema", str(path), "--context", context, "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "context" in captured.err


def test_cli_malformed_json_is_clean_error(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    assert cli.main(["validate", str(path)]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("error,code", [(ConcurrencyError("busy"), 1),
    (RuntimeError("inference failed"), 1), (ImportError("backend missing"), 1),
    (ValueError("bad input"), 2), (OSError("file unavailable"), 2)])
def test_cli_failure_exit_codes(tmp_path, monkeypatch, capsys, error, code):
    path = tmp_path / "schema.json"
    Schema(FIELDS).to_json(str(path))

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(cli, "Decider", fail)
    assert cli.main(["decide", "--schema", str(path), "--context", "x", "--json"]) == code
    captured = capsys.readouterr()
    assert captured.out == "" and str(error) in captured.err


@pytest.mark.parametrize("source", ["file", "stdin"])
def test_cli_context_sources(tmp_path, monkeypatch, source):
    import io
    from argparse import Namespace

    path = tmp_path / "context.txt"
    path.write_text("Ticket: café", encoding="utf-8")
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("Ticket: café"))
    args = Namespace(context=None, context_file=str(path) if source == "file" else None)
    assert cli._read_context(args) == "Ticket: café"


def test_python_config_precedence_and_calibration(clean_config, tmp_path, monkeypatch):
    from parallel_decisions import Calibrator

    calibration = tmp_path / "calibration.json"
    Calibrator().to_json(str(calibration))
    monkeypatch.setenv("PD_MODEL", "env/model")
    monkeypatch.setenv("PD_CALIBRATION", str(calibration))
    d = Decider(config=str(clean_config))
    assert d.model_id == "env/model"
    assert d.calibrator is not None and d._model is None
    assert d.cuda_graph is False
    explicit = Decider(config=str(clean_config), model_id="explicit/model")
    assert explicit.model_id == "explicit/model"


def test_integration_guide_python_blocks_parse():
    import ast
    import re

    text = (ROOT / "INTEGRATION.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    for block in blocks:
        ast.parse(block)


def test_python_result_json_preserves_types():
    values = {name: FieldValue(name, value, 0.9, []) for name, value in
              (("flag", False), ("category", "REVIEW"), ("actions", ["escalate"]))}
    response = DecisionResult(values)
    assert json.loads(json.dumps(response.json())) == {
        "flag": False, "category": "REVIEW", "actions": ["escalate"]}
    assert response.full_json()["flag"] == {"value": False, "probability": 0.9}


@pytest.mark.parametrize("timeout", ["-1", "nan", "inf"])
def test_server_rejects_invalid_timeout_before_load(serve, clean_config, monkeypatch, timeout):
    monkeypatch.setattr(serve, "Decider", lambda **kwargs: pytest.fail("must not load"))
    with pytest.raises(SystemExit) as exc:
        serve.main(["--config", str(clean_config), "--queue-timeout", timeout])
    assert exc.value.code == 2


def test_mcp_dict_schema_and_json(decider, monkeypatch):
    mcp = load_example("mcp_server")
    monkeypatch.setattr(mcp, "DECIDER", decider)
    for schema in (FIELDS, {"fields": FIELDS}):
        payload = json.loads(mcp.decide_tool("x", schema))
        assert payload["decisions"] == result().full_json()
        assert payload["calibrated"] is False
    assert "error" in json.loads(mcp.decide_tool("", FIELDS))
