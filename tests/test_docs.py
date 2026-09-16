"""The README's claims must be executable.

README snippets are the first thing a user copies. This parses every ```python block
for syntax and checks the shell blocks mention real commands, so a stale or broken
snippet fails the suite instead of the user.
"""

from __future__ import annotations

import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")


def readme() -> str:
    with open(README, encoding="utf-8") as fh:
        return fh.read()


def test_readme_python_snippets_parse():
    text = readme()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, "README has no python snippets to check"
    for i, block in enumerate(blocks, 1):
        try:
            ast.parse(block)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"README python block {i} does not parse: {exc}\n{block[:300]}")


def test_readme_commands_exist():
    """Every `pd <subcommand>` and `python <path>` in the README must be real."""
    text = readme()
    subcommands = set(re.findall(r"\.venv/bin/pd\s+([a-z]+)", text))
    assert subcommands <= {"validate", "decide", "calibrate", "config"}, subcommands
    for path in set(re.findall(r"python\s+([\w/\.-]+\.py)", text)):
        if path.startswith("smoke_test"):
            continue
        assert os.path.isfile(os.path.join(ROOT, path)), f"README refers to missing {path}"


def test_readme_mentions_every_cli_subcommand():
    from parallel_decisions.cli import main  # noqa: F401

    text = readme()
    for command in ("validate", "decide", "calibrate", "config"):
        assert f"pd {command}" in text, f"README does not document `pd {command}`"


def test_readme_documents_the_pd_toml_keys():
    """Config keys must be discoverable without reading the source."""
    from dataclasses import fields

    from parallel_decisions.config import Config

    text = readme()
    for field in fields(Config):
        if field.name == "source":
            continue
        assert field.name in text, f"README does not document the `{field.name}` setting"


def test_changelog_top_section_matches_package_version():
    import parallel_decisions

    with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as fh:
        text = fh.read()
    # first "## x.y.z" heading should be the current version
    match = re.search(r"^## (\d+\.\d+\.\d+)", text, re.M)
    assert match, "CHANGELOG has no version heading"
    assert match.group(1) == parallel_decisions.__version__
