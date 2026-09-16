"""Optional configuration: `pd.toml`, environment variables, clear precedence.

Precedence (highest first):

1. explicit argument to the library / CLI flag
2. environment variable (`PD_MODEL`, `PD_CALIBRATION`, ...)
3. `pd.toml` in the current directory, then `~/.config/parallel-decisions/pd.toml`
4. the built-in default

`pd.toml` is flat and small on purpose::

    model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    calibration = "~/.config/parallel-decisions/calibration.json"
    max_fields_per_batch = 6
    memory_budget_gb = 6.0
    warmup = true
    log = "json"            # or "off"
    lock_timeout_s = 0      # 0 = fail fast if another thread is mid-call
    backend = "auto"        # "auto" | "mlx" | "torch"
    torch_dtype = "bfloat16"  # torch backend: bfloat16 | float16 | float32
    torch_device = "cuda"     # torch backend: "cuda" | "cpu" (default: auto)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    tomllib = None  # type: ignore[assignment]

ENV_PREFIX = "PD_"
FILENAME = "pd.toml"
ENV_NAMES = {
    "model": "PD_MODEL",
    "calibration": "PD_CALIBRATION",
    "max_fields_per_batch": "PD_MAX_FIELDS_PER_BATCH",
    "memory_budget_gb": "PD_MEMORY_BUDGET_GB",
    "max_collision_rows": "PD_MAX_COLLISION_ROWS",
    "warmup": "PD_WARMUP",
    "log": "PD_LOG",
    "lock_timeout_s": "PD_LOCK_TIMEOUT_S",
    "backend": "PD_BACKEND",
    "torch_dtype": "PD_TORCH_DTYPE",
    "torch_device": "PD_TORCH_DEVICE",
    "cuda_graph": "PD_CUDA_GRAPH",
}
ENV_CONFIG = "PD_CONFIG"
BOOL_TRUE = {"1", "true", "yes", "on"}


class ConfigError(ValueError):
    """Raised for a malformed pd.toml."""


@dataclass
class Config:
    """Everything the engine accepts, all optional."""

    model: str | None = None
    calibration: str | None = None
    max_fields_per_batch: int | None = None
    memory_budget_gb: float | None = None
    max_collision_rows: int | None = None
    warmup: bool | None = None
    log: str | None = None
    lock_timeout_s: float | None = None
    backend: str | None = None
    torch_dtype: str | None = None
    torch_device: str | None = None
    cuda_graph: bool | None = None
    source: str | None = None

    def resolved(self) -> dict[str, Any]:
        """The set fields as a plain dict, without the bookkeeping `source`."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name != "source" and getattr(self, f.name) is not None}


def find_config_file(explicit: str | None = None, cwd: str | None = None) -> str | None:
    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.isfile(path):
            raise ConfigError(f"config file not found: {path}")
        return path
    env = os.environ.get(ENV_CONFIG)
    if env:
        return find_config_file(env, cwd)
    candidates = [
        os.path.join(cwd or os.getcwd(), FILENAME),
        os.path.expanduser(os.path.join("~", ".config", "parallel-decisions", FILENAME)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_config(path: str | None = None, cwd: str | None = None) -> Config:
    """Load `pd.toml` (if any) and overlay environment variables."""
    config = Config()
    found = find_config_file(path, cwd)
    if found:
        if tomllib is None:  # pragma: no cover - 3.10 fallback
            raise ConfigError("reading pd.toml needs Python 3.11+ (tomllib)")
        with open(found, "rb") as fh:
            try:
                data = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:  # type: ignore[attr-defined]
                raise ConfigError(f"{found}: {exc}") from exc
        config = _from_mapping(data, source=found)
    return _apply_env(config)


def _from_mapping(data: dict[str, Any], source: str | None = None) -> Config:
    known = {f.name for f in fields(Config)}
    # allow a single [parallel_decisions] / [pd] table wrapper
    for wrapper in ("parallel_decisions", "pd", "decider"):
        if wrapper in data and isinstance(data[wrapper], dict):
            data = {**data, **data[wrapper]}
    unknown = set(data) - known - {"parallel_decisions", "pd", "decider"}
    if unknown:
        raise ConfigError(f"unknown key(s) in pd.toml: {', '.join(sorted(unknown))}")
    config = Config(source=source)
    for key, value in data.items():
        if key not in known or value is None:
            continue
        try:
            setattr(config, key, _coerce(key, value))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"bad value for {key!r}: {value!r} ({exc})") from exc
    return config


def _coerce(key: str, value: Any) -> Any:
    if key in ("max_fields_per_batch", "max_collision_rows"):
        return int(value)
    if key in ("memory_budget_gb", "lock_timeout_s"):
        return float(value)
    if key in ("warmup", "cuda_graph"):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in BOOL_TRUE
    if key in ("model", "calibration", "log"):
        return str(value)
    return value


def _apply_env(config: Config) -> Config:
    for key, env_name in ENV_NAMES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        setattr(config, key, _coerce(key, raw))
    return config


def config_path_from_env() -> str | None:
    return os.environ.get("PD_CONFIG") or None
