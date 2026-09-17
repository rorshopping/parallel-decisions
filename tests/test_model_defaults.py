"""Backend-compatible model defaults, without loading a runtime or model."""

import builtins
import os

import pytest

import parallel_decisions
from parallel_decisions import Config, Decider
from parallel_decisions import engine
from parallel_decisions.config import ENV_NAMES
from parallel_decisions.engine_torch import DEFAULT_TORCH_MODEL


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    for name in (*ENV_NAMES.values(), "PD_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "pd.toml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("PD_CONFIG", str(path))
    # Construction must remain useful even if neither backend is installed.
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "mlx_lm", "torch", "transformers"}:
            raise AssertionError(f"constructor imported runtime dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)


@pytest.mark.parametrize("system,machine,backend,model", [
    ("win32", "AMD64", "torch", DEFAULT_TORCH_MODEL),
    ("win32", "ARM64", "torch", DEFAULT_TORCH_MODEL),
    ("linux", "x86_64", "torch", DEFAULT_TORCH_MODEL),
    ("linux", "aarch64", "torch", DEFAULT_TORCH_MODEL),
    ("darwin", "x86_64", "torch", DEFAULT_TORCH_MODEL),
    ("darwin", "arm64", "mlx", engine.DEFAULT_MODEL),
    ("darwin", "aarch64", "mlx", engine.DEFAULT_MODEL),
])
def test_auto_backend_default(monkeypatch, system, machine, backend, model):
    monkeypatch.setattr(engine.sys, "platform", system)
    monkeypatch.setattr(engine.platform, "machine", lambda: machine)
    decider = Decider()
    assert (decider.backend, decider.model_id) == (backend, model)
    assert decider._model is None
    assert decider._tokenizer is None
    assert decider._torch_rt is None


@pytest.mark.parametrize("backend,model", [
    ("torch", DEFAULT_TORCH_MODEL), ("mlx", engine.DEFAULT_MODEL),
])
@pytest.mark.parametrize("source", ["argument", "config", "environment", "file"])
def test_default_follows_resolved_backend(monkeypatch, backend, model, source):
    kwargs = {}
    if source == "argument":
        kwargs = {"backend": backend, "config": Config(backend="mlx" if backend == "torch" else "torch")}
    elif source == "config":
        kwargs = {"config": Config(backend=backend)}
    elif source == "environment":
        monkeypatch.setenv("PD_BACKEND", backend)
    else:
        with open(os.environ["PD_CONFIG"], "w", encoding="utf-8") as fh:
            fh.write(f'backend = "{backend}"\n')
    decider = Decider(**kwargs)
    assert (decider.backend, decider.model_id) == (backend, model)


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_model_precedence_is_argument_then_env_then_file(monkeypatch, backend):
    path = os.environ["PD_CONFIG"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('model = "file/model"\n')
    assert Decider(backend=backend).model_id == "file/model"
    monkeypatch.setenv("PD_MODEL", "env/model")
    assert Decider(backend=backend).model_id == "env/model"
    assert Decider("argument/model", backend=backend).model_id == "argument/model"


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_config_object_remains_pre_resolved(monkeypatch, backend):
    monkeypatch.setenv("PD_MODEL", "env/model")
    cfg = Config(model="config/model", backend=backend)
    assert Decider(config=cfg).model_id == "config/model"
    assert Decider("argument/model", config=cfg).model_id == "argument/model"


def test_legacy_default_model_remains_explicitly_usable():
    assert parallel_decisions.DEFAULT_MODEL == engine.DEFAULT_MODEL == \
        "mlx-community/Qwen2.5-7B-Instruct-4bit"
    # Explicit/configured IDs are not rewritten, even if incompatible with a backend.
    assert Decider(parallel_decisions.DEFAULT_MODEL, backend="torch").model_id == engine.DEFAULT_MODEL
    assert Decider(config=Config(model=engine.DEFAULT_MODEL, backend="torch")).model_id == engine.DEFAULT_MODEL
