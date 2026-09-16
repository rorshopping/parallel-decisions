"""Schema definitions for parallel-decisions.

Schema and Field: user-facing definitions plus tokenizer compilation.

A Schema maps field names to Field objects. Each Field is one of:
  - boolean: allowed answers true/false
  - enum:    allowed answers from a fixed choice list (2..255)
  - multi:   any subset of the choices, decided as one independent yes/no
             question per choice in the same batched pass

compile() converts each field into the pieces the engine needs:
  suffix tokens   e.g. '  "risk": "' or '  "risk": "common_prefix'
  candidate ids   first token of each allowed answer's remainder
  sequences       full token sequences (used when two answers share a first token)
  collision flag  set when two answers share a first token

A `multi` field expands into one CompiledField per choice (`choice_index` is set),
each with its own suffix like '  "actions[2]": ' so every choice gets a distinct
decision position while still landing in the same forward pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

MAX_CHOICES = 255


class SchemaError(ValueError):
    """Raised for malformed schemas."""


@dataclass
class Field:
    name: str
    type: str                      # "boolean" | "enum" | "multi"
    description: str = ""
    choices: list[str] = field(default_factory=list)
    choice_descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def is_boolean(self) -> bool:
        return self.type == "boolean"

    @property
    def is_multi(self) -> bool:
        return self.type == "multi"

    @property
    def answers(self) -> list[str]:
        """The literal strings the model chooses between at a decision position."""
        if self.is_multi:
            return ["true", "false"]   # one yes/no per choice
        return ["true", "false"] if self.is_boolean else list(self.choices)


class Schema:
    def __init__(self, fields: Mapping[str, Mapping[str, Any]] | Sequence[Field]):
        if isinstance(fields, Mapping):
            parsed: dict[str, Field] = {}
            for name, spec in fields.items():
                if not isinstance(spec, Mapping):
                    raise SchemaError(f"field {name!r}: expected an object, got {type(spec).__name__}")
                ftype = str(spec.get("type", "enum")).lower()
                if ftype not in ("boolean", "enum", "choice", "selection", "multi", "multi_select"):
                    raise SchemaError(
                        f"field {name!r}: unsupported type {ftype!r} "
                        f"(use 'boolean', 'enum' or 'multi')")
                if ftype in ("choice", "selection"):
                    ftype = "enum"
                if ftype == "multi_select":
                    ftype = "multi"
                choices: list[str] = []
                choice_descriptions: dict[str, str] = {}
                if ftype == "boolean":
                    choices = ["true", "false"]
                    raw = spec.get("choices")
                    if isinstance(raw, Mapping):
                        # {true: "...", false: "..."} — the definitions matter as much
                        # for booleans as for enums (the reference eval passes them)
                        choice_descriptions = {str(k).lower(): str(v) for k, v in raw.items()}
                else:
                    raw = spec.get("choices")
                    if isinstance(raw, Mapping):
                        # {choice: description} form, like TypeSafe's criteria
                        choice_descriptions = {str(k): str(v) for k, v in raw.items()}
                        choices = list(choice_descriptions.keys())
                    elif isinstance(raw, (list, tuple)):
                        choices = [str(c) for c in raw]
                    else:
                        raise SchemaError(f"field {name!r}: enum/multi fields require a 'choices' list or mapping")
                    if len(choices) < 2:
                        raise SchemaError(f"field {name!r}: needs at least 2 choices")
                    if len(choices) > MAX_CHOICES:
                        raise SchemaError(f"field {name!r}: {len(choices)} choices exceeds the {MAX_CHOICES} limit")
                    if len(set(choices)) != len(choices):
                        raise SchemaError(f"field {name!r}: duplicate choices")
                parsed[str(name)] = Field(
                    name=str(name),
                    type=ftype,
                    description=str(spec.get("description", "")).strip(),
                    choices=choices,
                    choice_descriptions=choice_descriptions,
                )
        else:
            parsed = {f.name: f for f in fields}
        if not parsed:
            raise SchemaError("schema is empty")
        self.fields: dict[str, Field] = parsed

    # ---- convenience -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def __iter__(self):
        return iter(self.fields)

    def to_list(self) -> list[dict[str, Any]]:
        out = []
        for f in self.fields.values():
            item: dict[str, Any] = {"name": f.name, "type": f.type, "description": f.description}
            if not f.is_boolean:
                item["choices"] = dict(f.choice_descriptions) if f.choice_descriptions else list(f.choices)
            out.append(item)
        return out
    # ---- (de)serialisation -------------------------------------------------
    def to_json(self, path: str | None = None) -> str:
        text = json.dumps({"fields": self.to_list()}, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_text: str) -> "Schema":
        if os.path.exists(path_or_text):
            with open(path_or_text, encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(path_or_text)
        fields = data.get("fields", data)
        if isinstance(fields, list):
            return cls({item["name"]: item for item in fields})
        return cls(fields)

    # ---- tokenizer compilation --------------------------------------------
    def compile(self, tokenizer) -> list["CompiledField"]:
        """Compile each field for the engine. Requires an mlx_lm tokenizer.

        A `multi` field compiles to one boolean row per choice, keyed
        `"<name>[<index>]"` so each choice gets its own decision position.
        """
        out: list[CompiledField] = []
        for f in self.fields.values():
            if f.is_multi:
                for i in range(len(f.choices)):
                    out.append(CompiledField.build(f, tokenizer, choice_index=i))
            else:
                out.append(CompiledField.build(f, tokenizer))
        return out


@dataclass
class CompiledField:
    field: Field
    suffix: str                        # the literal text before the answer
    suffix_tokens: list[int]           # its token ids
    candidate_ids: list[list[int]]     # the token that starts each allowed answer
    sequences: list[list[int]]         # tokens of each answer as actually emitted
    collision: bool                    # two answers start with the same token
    choice_index: int | None = None    # set for one row of a multi-select field

    def __post_init__(self) -> None:
        if self.choice_index is not None and not self.field.is_multi:
            raise SchemaError(f"field {self.field.name!r}: choice_index only applies to multi fields")

    @property
    def row_name(self) -> str:
        """The literal key this row's suffix writes (multi rows are indexed)."""
        if self.choice_index is None:
            return self.field.name
        return f"{self.field.name}[{self.choice_index}]"

    @staticmethod
    def _next_token(tokenizer, suffix: str, suffix_tokens: Sequence[int], answer: str) -> tuple[int, list[int]]:
        """The token the model emits right after `suffix` when the answer is `answer`.

        Read off the real encoding of `suffix + answer`, which is correct whenever the
        tokenizer keeps the suffix intact (`encode(suffix)` is a token-prefix of
        `encode(suffix + answer)`).

        It usually does not. Every boolean row in the reference evaluation looks like
        `  "name": ` whose last token is a bare space; re-encoding that together with
        `true` merges the space into ` true`, a token the model cannot emit because it
        has already emitted the space. In that case the answer's own first token is the
        right candidate, which is also what the reference implementation scored. Guessing
        wrong here re-weights an entire field, so this is pinned by tests.

        Returns (first token id, the tokens of the answer as it would be emitted).
        """
        full = [int(t) for t in tokenizer.encode(suffix + answer, add_special_tokens=False)]
        n = len(suffix_tokens)
        if len(full) > n and list(full[:n]) == list(suffix_tokens):
            return full[n], full[n:]
        # The tokenizer merged across the suffix/answer boundary; fall back to the
        # answer's own encoding so the row still has a usable decision position.
        bare = [int(t) for t in tokenizer.encode(answer, add_special_tokens=False)]
        if not bare:
            return -1, [0]
        return bare[0], bare

    @classmethod
    def build(cls, f: Field, tokenizer, choice_index: int | None = None) -> "CompiledField":
        if f.is_boolean or f.is_multi:
            key = f.name if choice_index is None else f"{f.name}[{choice_index}]"
            suffix = f'  "{key}": '
            suffix_tokens = [int(t) for t in tokenizer.encode(suffix, add_special_tokens=False)]
            candidates: list[list[int]] = []
            sequences: list[list[int]] = []
            firsts: list[int] = []
            for answer in ("true", "false"):
                first, seq = cls._next_token(tokenizer, suffix, suffix_tokens, answer)
                firsts.append(first)
                candidates.append([first])
                sequences.append(seq)
        else:
            prefix = os.path.commonprefix(f.choices)
            suffix = f'  "{f.name}": "{prefix}'
            suffix_tokens = [int(t) for t in tokenizer.encode(suffix, add_special_tokens=False)]
            candidates = []
            sequences = []
            firsts = []
            for choice in f.choices:
                first, seq = cls._next_token(tokenizer, suffix, suffix_tokens, choice[len(prefix):])
                firsts.append(first)
                candidates.append([first])
                sequences.append(seq)

        collision = len(set(firsts)) < len(firsts)
        return cls(
            field=f,
            suffix=suffix,
            suffix_tokens=suffix_tokens,
            candidate_ids=candidates,
            sequences=sequences,
            collision=collision,
            choice_index=choice_index,
        )

    def candidate_ids_for(self, answer_index: int) -> list[int]:
        return self.candidate_ids[answer_index]
