"""Config, concurrency and platform-guard tests. No model, no MLX needed."""

from __future__ import annotations

import sys

import pytest

from parallel_decisions.config import Config, ConfigError, find_config_file, load_config


def test_load_config_from_toml(tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text(
        'model = "some/model"\n'
        "max_fields_per_batch = 6\n"
        "memory_budget_gb = 3.5\n"
        "warmup = false\n"
        'log = "json"\n',
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.model == "some/model"
    assert cfg.max_fields_per_batch == 6
    assert cfg.memory_budget_gb == 3.5
    assert cfg.warmup is False
    assert cfg.log == "json"
    assert cfg.source == str(path)


def test_env_overrides_toml(tmp_path, monkeypatch):
    path = tmp_path / "pd.toml"
    path.write_text('model = "from/file"\n', encoding="utf-8")
    monkeypatch.setenv("PD_MODEL", "from/env")
    monkeypatch.setenv("PD_MEMORY_BUDGET_GB", "2.5")
    cfg = load_config(str(path))
    assert cfg.model == "from/env"
    assert cfg.memory_budget_gb == 2.5


def test_find_config_prefers_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("PD_CONFIG", raising=False)
    (tmp_path / "pd.toml").write_text('model = "cwd/model"\n', encoding="utf-8")
    found = find_config_file(cwd=str(tmp_path))
    assert found == str(tmp_path / "pd.toml")


def test_missing_config_file_is_an_error():
    with pytest.raises(ConfigError):
        find_config_file("/nonexistent/pd.toml")


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text('nonsense = 3\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_bad_value_type_is_rejected(tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text('max_fields_per_batch = "many"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_wrapped_table_is_accepted(tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text('[parallel_decisions]\nmodel = "wrapped/model"\n', encoding="utf-8")
    assert load_config(str(path)).model == "wrapped/model"


def test_resolved_excludes_unset_fields_and_source(tmp_path, monkeypatch):
    monkeypatch.delenv("PD_MODEL", raising=False)
    path = tmp_path / "pd.toml"
    path.write_text('model = "m"\n', encoding="utf-8")
    assert load_config(str(path)).resolved() == {"model": "m"}


def test_empty_config_is_fine():
    assert Config().resolved() == {}


def test_env_bool_variants(monkeypatch, tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text("", encoding="utf-8")
    for raw, expected in (("1", True), ("true", True), ("no", False), ("OFF", False)):
        monkeypatch.setenv("PD_WARMUP", raw)
        assert load_config(str(path)).warmup is expected


def test_decider_takes_settings_from_config_without_loading_a_model():
    """Decider must resolve config in __init__ and only load lazily."""
    from parallel_decisions.engine import Decider

    cfg = Config(model="cfg/model", max_fields_per_batch=3, memory_budget_gb=2.0,
                 max_collision_rows=4, warmup=False, lock_timeout_s=0.05, log="json")
    decider = Decider(config=cfg)
    assert decider.model_id == "cfg/model"
    assert decider.max_fields_per_batch == 3
    assert decider.memory_budget_bytes == int(2.0 * 1024 ** 3)
    assert decider.max_collision_rows == 4
    assert decider.warmup is False
    assert decider.lock_timeout_s == 0.05
    assert decider.log_mode == "json"
    assert decider._model is None      # nothing loaded yet


def test_explicit_argument_beats_config():
    from parallel_decisions.engine import Decider

    cfg = Config(model="cfg/model", max_fields_per_batch=3)
    decider = Decider(model_id="explicit/model", max_fields_per_batch=9, config=cfg,
                      warmup=False)
    assert decider.model_id == "explicit/model"
    assert decider.max_fields_per_batch == 9


def test_lock_times_out_with_a_clear_error():
    from parallel_decisions.engine import ConcurrencyError, Decider

    decider = Decider(model_id="unused", warmup=False, lock_timeout_s=0.05,
                      config=Config())
    assert decider._lock.acquire(timeout=1.0)
    try:
        with pytest.raises(ConcurrencyError):
            decider._acquire_lock("decide")
    finally:
        decider._lock.release()


def test_lock_is_released_after_use():
    from parallel_decisions.engine import Decider

    decider = Decider(model_id="unused", warmup=False, lock_timeout_s=0.0, config=Config())
    for _ in range(3):
        with decider._acquire_lock("test"):
            pass
    assert decider._lock.acquire(timeout=0.1)
    decider._lock.release()


def test_platform_guard_rejects_non_arm(monkeypatch):
    from parallel_decisions.engine import Decider, UnsupportedPlatformError

    monkeypatch.delenv("PD_ALLOW_NON_ARM", raising=False)
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    # the torch backend intentionally works on non-arm64 hosts; only the MLX
    # backend is Apple-Silicon-only, so force it for this test
    decider = Decider(model_id="unused", warmup=False, config=Config(),
                      backend="mlx")
    with pytest.raises(UnsupportedPlatformError):
        decider._check_platform()


def test_auto_backend_selects_torch_off_mac(monkeypatch):
    from parallel_decisions.engine import _select_backend

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert _select_backend("auto") == "torch"


def test_auto_backend_selects_mlx_on_apple_silicon(monkeypatch):
    from parallel_decisions.engine import _select_backend

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert _select_backend("auto") == "mlx"


def test_backend_argument_overrides_config(monkeypatch):
    from parallel_decisions.engine import Decider

    decider = Decider(model_id="unused", warmup=False, config=Config(),
                      backend="torch")
    assert decider.backend == "torch"


def test_backend_rejects_unknown_values():
    from parallel_decisions.engine import _select_backend

    with pytest.raises(ValueError):
        _select_backend("tensorrt")


def test_platform_guard_can_be_overridden(monkeypatch):
    from parallel_decisions.engine import Decider

    monkeypatch.setenv("PD_ALLOW_NON_ARM", "1")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    Decider(model_id="unused", warmup=False, config=Config())._check_platform()


def test_log_event_is_json_on_stderr(capsys, monkeypatch):
    from parallel_decisions.engine import Decider

    monkeypatch.delenv("PD_LOG", raising=False)
    decider = Decider(model_id="m", warmup=False, config=Config(log="json"))
    decider.log_event("decide", fields=3, chunks=1)
    err = capsys.readouterr().err.strip()
    import json

    payload = json.loads(err)
    assert payload == {"event": "decide", "model": "m", "fields": 3, "chunks": 1}


def test_log_event_silent_by_default(capsys, monkeypatch):
    from parallel_decisions.engine import Decider

    monkeypatch.delenv("PD_LOG", raising=False)
    Decider(model_id="m", warmup=False, config=Config()).log_event("decide", fields=1)
    assert capsys.readouterr().err == ""
