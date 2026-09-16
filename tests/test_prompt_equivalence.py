"""Prompt-equivalence test against the research runner.

The accuracy numbers quoted in README.md/CALIBRATION.md were measured with the
research tree's runner (`rlcd-research/source/Qwen-2.5-1B-RLCD/core/engine_mlx.py`).
This test rebuilds the same prompt text here from the published criteria and fails
if the packaged `build_prompt` diverges — a silent prompt change would invalidate
every number in the READMEs.

It runs only when the research tree is present (it is not vendored into the package).
"""

from __future__ import annotations

import json
import os

import pytest

from parallel_decisions.prompts import build_prompt
from parallel_decisions.schema import Schema

FULL_EVAL = os.path.expanduser(
    "~/Documents/projects/rlcd-research/evals/full_eval.json")


def research_schema_dict(questions):
    """Byte-for-byte copy of `eval_local_chunked.build_schema_dict`.

    Some published noul questions have `criteria: null`; the chunked runner that
    produced the recorded numbers falls back to the bare instructions there, so this
    copy does the same.
    """
    schema = {}
    for q in questions:
        crit = q["criteria"]
        if q["type"] == "noul":
            if isinstance(crit, dict):
                desc = (f"{q['instructions']} (true: {crit.get('true', '')}; "
                        f"false: {crit.get('false', '')})")
            else:
                desc = q["instructions"]
            schema[q["qid"]] = {"type": "boolean", "description": desc}
        elif q["type"] == "score":
            levels = "; ".join(f"{i}={v}" for i, v in enumerate(crit)) if isinstance(crit, list) else ""
            schema[q["qid"]] = {"type": "enum", "choices": ["0", "1", "2", "3"],
                                "description": f"{q['instructions']} Levels: {levels}"}
        else:
            if isinstance(crit, dict):
                opts = "; ".join(f"{k}={v}" for k, v in crit.items())
                choices = list(crit.keys())
            else:
                opts, choices = "", []
            schema[q["qid"]] = {"type": "enum", "choices": choices,
                                "description": f"{q['instructions']} Options: {opts}"}
    return schema


def research_engine_prompt(context, schema_dict):
    """The research engine's `base_prompt`, minus its f-string interpolation."""
    lines = [f'  "{name}": {spec["description"]}' for name, spec in schema_dict.items()]
    schema_str = "\n".join(lines)
    return (
        f"<|im_start|>system\n"
        f"Classify JSON attributes:\n{schema_str}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n"
    )


needs_tree = pytest.mark.skipif(
    not os.path.isfile(FULL_EVAL),
    reason="research tree not present (expected at ~/Documents/projects/rlcd-research)")


@needs_tree
def test_prompt_matches_research_runner_on_published_cases():
    """Every published question's prompt must be byte-identical to the runner's."""
    full = json.load(open(FULL_EVAL, encoding="utf-8"))
    checked = 0
    for wf, wdata in full["workflows"].items():
        for case in wdata["cases"]:
            for question in case["questions"]:
                # one-question schema: enough to compare the field line exactly
                original = research_schema_dict([question])
                packaged = Schema(original)
                assert build_prompt(case["input_text"], packaged) == \
                    research_engine_prompt(case["input_text"], original), \
                    f"prompt drift for {wf}/{case['case_id']}/{question['qid']}"
                checked += 1
    assert checked == 373, f"expected 373 question slots, checked {checked}"


@needs_tree
def test_choice_descriptions_produce_the_reference_option_format():
    """A schema written with the {choice: description} form renders as 'k=v; k=v'."""
    schema = Schema({"f": {"type": "enum", "description": "pick",
                          "choices": {"a": "first", "b": "second"}}})
    prompt = build_prompt("ctx", schema)
    assert '  "f": pick Options: a=first; b=second' in prompt


@needs_tree
def test_boolean_definitions_reach_the_prompt():
    schema = Schema({"f": {"type": "boolean", "description": "is it so",
                           "choices": {"true": "yes it is", "false": "no it is not"}}})
    prompt = build_prompt("ctx", schema)
    assert "is it so (true: yes it is; false: no it is not)" in prompt


def test_boolean_without_definitions_keeps_the_plain_line():
    schema = Schema({"f": {"type": "boolean", "description": "is it so"}})
    assert '  "f": is it so' in build_prompt("ctx", schema)


def test_enum_without_descriptions_keeps_the_plain_option_list():
    schema = Schema({"f": {"type": "enum", "description": "pick", "choices": ["a", "b"]}})
    assert '  "f": pick Options: a; b' in build_prompt("ctx", schema)
