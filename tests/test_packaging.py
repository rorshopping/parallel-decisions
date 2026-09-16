"""Packaging tests: the version has one source of truth and the wheel is sane.

These run without a model and without the package being installed from a wheel;
`test_metadata_version_matches_package` is skipped when the metadata is absent.
"""

from __future__ import annotations

import os
import tomllib

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_version_is_dynamic_in_pyproject():
    """The version must come from the package, not from a second literal.

    A hardcoded `version` in pyproject.toml silently drifts from `__version__`;
    that is what happened at 0.2.0 (the wheel was built as 0.1.0).
    """
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        data = tomllib.load(fh)
    assert "version" not in data["project"], "version must not be hardcoded"
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == \
        "parallel_decisions.__version__"


def test_package_version_is_a_release_version():
    import parallel_decisions

    parts = parallel_decisions.__version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_metadata_version_matches_package():
    """When built/installed, the distribution version equals `__version__`."""
    from importlib.metadata import PackageNotFoundError, version

    import parallel_decisions

    try:
        installed = version("parallel-decisions")
    except PackageNotFoundError:
        pytest.skip("package metadata not available (not installed)")
    assert installed == parallel_decisions.__version__


def test_changelog_documents_the_current_version():
    import parallel_decisions

    with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert f"## {parallel_decisions.__version__}" in text, \
        "CHANGELOG.md must have a section for the current version"


def test_public_api_is_importable_without_mlx_installed(monkeypatch):
    """`import parallel_decisions` must not require MLX; only a call does.

    The console script and the schema/calibration helpers are useful on a machine
    that will never run the model, and errors should be actionable, not ImportError
    at import time.
    """
    import subprocess
    import sys

    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] == 'mlx':\n"
        "        raise ImportError('no mlx in this test')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import parallel_decisions as pd\n"
        "assert pd.__version__\n"
        "assert pd.Calibrator().kind == 'identity'\n"
        "from parallel_decisions.schema import Schema\n"
        "Schema({'a': {'type': 'boolean', 'description': 'x'}})\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
