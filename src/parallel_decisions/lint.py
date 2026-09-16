"""Schema lints: problems the tokenizer reveals, not the schema author.

`lint_schema()` reports the two things that actually hurt at runtime:

1. **Token collisions.** Two allowed answers that begin with the same token cannot
   be told apart by the single batched pass; the engine falls back to scoring the
   full continuations in a second pass. That is correct but slower, and it only
   works because the engine does it — a caller reading the probabilities should know
   the row needed the slow path.
2. **Fields the model must read past a shared token.** The rename advice below is
   the actionable part: if the distinguishing word comes *after* a shared prefix, it
   becomes part of the continuation score rather than the decision. Moving the
   distinguishing word first makes the field cheap and usually sharper.

Neither problem changes which answer wins, so this is a cost/clarity lint, not a
correctness gate. `has_blocking_issues()` is the one thing to hard-fail on:
answers that are literally the same string (Schema already rejects those) or
answers that tokenize identically end-to-end (they cannot be distinguished at all).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import CompiledField, Schema

__all__ = ["FieldLint", "SchemaLint", "lint_schema"]


@dataclass
class FieldLint:
    """One finding about one field."""

    name: str
    kind: str                     # "collision" | "identical" | "long_answer"
    detail: str
    groups: list[list[str]] = field(default_factory=list)   # choices sharing a token
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.name, "kind": self.kind, "detail": self.detail,
                "groups": self.groups, "suggestions": self.suggestions}


@dataclass
class SchemaLint:
    fields: list[FieldLint] = field(default_factory=list)
    rows: int = 0
    fields_total: int = 0

    @property
    def collisions(self) -> list[FieldLint]:
        return [f for f in self.fields if f.kind == "collision"]

    @property
    def blocking(self) -> list[FieldLint]:
        return [f for f in self.fields if f.kind == "identical"]

    def has_blocking_issues(self) -> bool:
        return bool(self.blocking)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields_total": self.fields_total,
            "decision_rows": self.rows,
            "colliding_fields": len(self.collisions),
            "blocking_fields": len(self.blocking),
            "findings": [f.to_dict() for f in self.fields],
        }


def _shared_token_groups(cf: CompiledField) -> list[list[str]]:
    """Choice groups that emit the same first token."""
    by_token: dict[int, list[str]] = {}
    for choice, ids in zip(cf.field.choices, cf.candidate_ids):
        by_token.setdefault(ids[0], []).append(choice)
    return [group for group in by_token.values() if len(group) > 1]


def _rename_suggestions(group: Sequence[str]) -> list[str]:
    """Concrete renames that give each choice a distinct first token.

    The usual cause is a shared prefix with the distinguishing word afterwards
    (`extra_approved` / `extra_unapproved`), so the advice is to move the first word
    that differs to the front: `approved_extra` / `unapproved_extra`. Splitting is on
    underscores only — a heuristic, and the suggestion is meant to be reviewed, not
    applied blindly.
    """
    words = [c.split("_") for c in group]
    for j in range(len(words[0])):
        if all(len(w) > j for w in words) and len({w[j] for w in words}) == len(words):
            if j == 0:
                return [f"{c} -> {c}_v2" for c in group]   # already distinct at word 0
            reordered = ["_".join(w[j:j + 1] + w[:j] + w[j + 1:]) for w in words]
            if len(set(reordered)) == len(reordered) and all(reordered):
                return [f"{a} -> {b}" for a, b in zip(group, reordered)]
            break
    return [f"{c} -> q_{c}" for c in group]


def lint_schema(schema: Schema, tokenizer) -> SchemaLint:
    """Check a schema against a tokenizer. Needs a loaded tokenizer, not a model."""
    compiled = schema.compile(tokenizer)
    rows = len(compiled)
    findings: list[FieldLint] = []

    for cf in compiled:
        name = cf.row_name
        if cf.field.is_boolean or cf.field.is_multi:
            continue
        # answers that cannot be distinguished at all: same tokens end to end
        seen: dict[tuple[int, ...], list[str]] = {}
        for choice, seq in zip(cf.field.choices, cf.sequences):
            seen.setdefault(tuple(seq), []).append(choice)
        for seq, choices in seen.items():
            if len(choices) > 1:
                findings.append(FieldLint(
                    name=name, kind="identical",
                    detail=(f"{choices} tokenize identically, so the engine cannot "
                            f"choose between them"),
                    groups=[choices],
                    suggestions=_rename_suggestions(choices),
                ))
        if not cf.collision:
            continue
        groups = _shared_token_groups(cf)
        suggestions: list[str] = []
        for group in groups:
            suggestions.extend(_rename_suggestions(group))
        findings.append(FieldLint(
            name=name, kind="collision",
            detail=(f"{len(cf.field.choices)} choices share a first token "
                    f"({len(groups)} shared-token group(s)); needs the extra "
                    f"sequence-scoring pass"),
            groups=groups,
            suggestions=suggestions,
        ))

    return SchemaLint(fields=findings, rows=rows, fields_total=len(schema))
