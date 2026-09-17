"""Packaging tests: the version has one source of truth and the wheel is sane.

These run without a model and without the package being installed from a wheel;
`test_metadata_version_matches_package` is skipped when the metadata is absent.
"""

from __future__ import annotations

import os
import tomllib

import pytest
from packaging.requirements import Requirement


@pytest.fixture
def project_metadata():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)["project"]


@pytest.mark.parametrize("system,machine,needs_mlx", [
    ("win32", "AMD64", False),
    ("win32", "ARM64", False),
    ("linux", "x86_64", False),
    ("linux", "aarch64", False),
    ("darwin", "x86_64", False),
    ("darwin", "arm64", True),
    ("darwin", "aarch64", True),
])
def test_base_dependencies_follow_platform(project_metadata, system, machine, needs_mlx):
    requirements = [Requirement(value) for value in project_metadata["dependencies"]]
    env = {"sys_platform": system, "platform_machine": machine, "extra": ""}
    active = {req.name for req in requirements
              if req.marker is None or req.marker.evaluate(env)}
    assert active == ({"mlx", "mlx-lm"} if needs_mlx else set())


def test_torch_extra_declares_supported_api_floor(project_metadata):
    requirements = {req.name: req for req in map(
        Requirement, project_metadata["optional-dependencies"]["torch"])}
    assert {"torch", "transformers", "accelerate"} <= requirements.keys()
    assert "1.1.0" in requirements["accelerate"].specifier
    assert "1.0.0" not in requirements["accelerate"].specifier
    # No platform markers: the extra is also usable on a Mac with backend="torch".
    assert all(req.marker is None for req in requirements.values())
    assert "2.6.0" in requirements["torch"].specifier
    assert "2.5.1" not in requirements["torch"].specifier
    assert "5.17.0" in requirements["transformers"].specifier
    assert "4.57.0" not in requirements["transformers"].specifier
    assert "5.16.0" not in requirements["transformers"].specifier
    assert "6.0.0" not in requirements["transformers"].specifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_built_metadata_preserves_dependency_markers(tmp_path, project_metadata):
    """Check emitted Requires-Dist, not another checkout's installed metadata."""
    import importlib.util
    import shutil
    import subprocess
    import sys
    from email.parser import Parser
    from pathlib import Path

    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools build backend unavailable")
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(Path(ROOT) / name, source / name)
    shutil.copytree(Path(ROOT) / "src" / "parallel_decisions",
                    source / "src" / "parallel_decisions", ignore=shutil.ignore_patterns("__pycache__"))
    output = source / "metadata"
    output.mkdir()
    proc = subprocess.run([
        sys.executable, "-c",
        "from setuptools.build_meta import prepare_metadata_for_build_wheel; "
        "prepare_metadata_for_build_wheel('metadata')",
    ], cwd=source, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    paths = list(output.glob("*.dist-info/METADATA"))
    assert len(paths) == 1
    metadata = Parser().parsestr(paths[0].read_text(encoding="utf-8"))
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    assert set(metadata.get_all("Provides-Extra", [])) == set(project_metadata["optional-dependencies"])
    for system, machine, extra, expected in [
        ("win32", "AMD64", "", set()),
        ("linux", "x86_64", "", set()),
        ("darwin", "x86_64", "", set()),
        ("darwin", "arm64", "", {"mlx", "mlx-lm"}),
        ("win32", "AMD64", "torch", {"torch", "transformers", "accelerate"}),
    ]:
        env = {"sys_platform": system, "platform_machine": machine, "extra": extra}
        active = {req.name for req in requirements
                  if req.marker is None or req.marker.evaluate(env)}
        assert active == expected


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


def test_public_api_is_importable_without_backends_installed():
    """Import and construction must not require either inference backend.

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
        "    if name.split('.')[0] in {'mlx', 'mlx_lm', 'torch', 'transformers'}:\n"
        "        raise ImportError('no backend dependencies in this test')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import parallel_decisions as pd\n"
        "assert pd.__version__\n"
        "assert pd.Calibrator().kind == 'identity'\n"
        "d = pd.Decider(backend='torch', config=pd.Config())\n"
        "assert d.model_id == 'Qwen/Qwen2.5-0.5B-Instruct'\n"
        "assert d._model is None\n"
        "from parallel_decisions.schema import Schema\n"
        "Schema({'a': {'type': 'boolean', 'description': 'x'}})\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
