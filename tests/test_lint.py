"""Schema lint tests: collisions, indistinguishable answers, rename advice.

Uses the same vocabulary-based stub tokenizer as the schema tests — no MLX, no model.
"""

from __future__ import annotations

from parallel_decisions.lint import lint_schema
from parallel_decisions.schema import Schema

from test_schema import _tok


class _CaseInsensitiveTokenizer:
    """A tokenizer that collapses case, so distinct strings become identical tokens."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text.endswith('": "'):
            return [90]
        word = text.split('"')[-1].strip().lower()
        return [1000 + sum(map(ord, word)) % 500]


def test_clean_schema_reports_nothing():
    schema = Schema({
        "risk": {"type": "enum", "choices": ["prorated", "extra"], "description": "x"},
        "flag": {"type": "boolean", "description": "x"},
    })
    report = lint_schema(schema, _tok())
    assert report.fields == []
    assert report.has_blocking_issues() is False
    assert report.fields_total == 2
    assert report.rows == 2


def test_collision_is_reported_with_the_shared_group():
    schema = Schema({"f": {"type": "enum",
                           "choices": ["extra_approved", "extra_unapproved", "prorated"],
                           "description": "x"}})
    report = lint_schema(schema, _tok())
    assert len(report.collisions) == 1
    finding = report.collisions[0]
    assert finding.name == "f"
    assert finding.groups == [["extra_approved", "extra_unapproved"]]
    assert report.has_blocking_issues() is False   # collisions are a cost, not a bug


def test_collision_suggestions_move_the_distinguishing_word_first():
    # a third choice with a different opening breaks the common prefix, which is
    # what makes the two `extra_` choices collide on their first token
    schema = Schema({"f": {"type": "enum",
                           "choices": ["extra_approved", "extra_unapproved", "prorated"],
                           "description": "x"}})
    finding = lint_schema(schema, _tok()).collisions[0]
    assert "extra_approved -> approved_extra" in finding.suggestions
    assert "extra_unapproved -> unapproved_extra" in finding.suggestions


def test_case_differing_choices_that_tokenize_alike_are_blocking():
    """Distinct strings that produce the same tokens cannot be told apart at all."""
    schema = Schema({"f": {"type": "enum", "choices": ["Alpha", "alpha"],
                           "description": "x"}})
    report = lint_schema(schema, _CaseInsensitiveTokenizer())
    assert report.has_blocking_issues() is True
    finding = report.blocking[0]
    assert finding.kind == "identical"
    assert finding.groups == [["Alpha", "alpha"]]
    assert finding.suggestions                        # something to rename to


def test_multi_and_boolean_rows_are_not_linted():
    schema = Schema({
        "flag": {"type": "boolean", "description": "x"},
        "actions": {"type": "multi", "choices": ["retry", "escalate"], "description": "x"},
    })
    report = lint_schema(schema, _tok())
    assert report.fields == []
    assert report.rows == 3          # 1 boolean + 2 multi rows
    assert report.fields_total == 2


def test_report_serialises_and_counts_match():
    schema = Schema({"f": {"type": "enum",
                           "choices": ["extra_approved", "extra_unapproved", "prorated"],
                           "description": "x"}})
    payload = lint_schema(schema, _tok()).to_dict()
    assert payload["colliding_fields"] == 1
    assert payload["blocking_fields"] == 0
    assert payload["decision_rows"] == 1
    assert payload["findings"][0]["kind"] == "collision"


def test_no_collision_when_choices_differ_at_the_first_token():
    schema = Schema({"f": {"type": "enum", "choices": ["prorated", "extra_approved"],
                           "description": "x"}})
    assert lint_schema(schema, _tok()).collisions == []
