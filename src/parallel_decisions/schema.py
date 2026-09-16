"""Schema definitions for parallel-decisions.

Schema and Field: user-facing definitions plus tokenizer compilation.

A Schema maps field names to Field objects. Each Field is either:
  - boolean: allowed answers true/false
  - enum:    allowed answers from a fixed choice list (2..255)

compile() converts each field into the pieces the engine needs:
  suffix tokens   e.g. '  "risk": "' or '  "risk": "common_prefix'
  candidate ids   first token of each allowed answer's remainder
  sequences       full token sequences (used when two answers share a first token)
  collision flag  set when two answers share a first token
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
    type: str                      # "boolean" | "enum"
    description: str = ""
    choices: list[str] = field(default_factory=list)
    choice_descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def is_boolean(self) -> bool:
        return self.type == "boolean"

    @property
    def answers(self) -> list[str]:
        return ["true", "false"] if self.is_boolean else list(self.choices)


class Schema:
    def __init__(self, fields: Mapping[str, Mapping[str, Any]] | Sequence[Field]):
        if isinstance(fields, Mapping):
            parsed: dict[str, Field] = {}
            for name, spec in fields.items():
                if not isinstance(spec, Mapping):
                    raise SchemaError(f"field {name!r}: expected an object, got {type(spec).__name__}")
                ftype = str(spec.get("type", "enum")).lower()
                if ftype not in ("boolean", "enum", "choice", "selection"):
                    raise SchemaError(f"field {name!r}: unsupported type {ftype!r} (use 'boolean' or 'enum')")
                if ftype in ("choice", "selection"):
                    ftype = "enum"
                choices: list[str] = []
                choice_descriptions: dict[str, str] = {}
                if ftype == "boolean":
                    choices = ["true", "false"]
                else:
                    raw = spec.get("choices")
                    if isinstance(raw, Mapping):
                        # {choice: description} form, like TypeSafe's criteria
                        choice_descriptions = {str(k): str(v) for k, v in raw.items()}
                        choices = list(choice_descriptions.keys())
                    elif isinstance(raw, (list, tuple)):
                        choices = [str(c) for c in raw]
                    else:
                        raise SchemaError(f"field {name!r}: enum fields require a 'choices' list or mapping")
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
        """Compile each field for the engine. Requires an mlx_lm tokenizer."""
        return [CompiledField.build(f, tokenizer) for f in self.fields.values()]


@dataclass
class CompiledField:
    field: Field
    suffix_tokens: list[int]           # tokens of the literal text before the answer
    candidate_ids: list[list[int]]     # candidate first-token ids per allowed answer
    sequences: list[list[int]]         # full token sequences of each answer remainder
    collision: bool                    # two answers share a first token

    @staticmethod
    def _first_tokens(tokenizer, text: str) -> list[int]:
        """Token ids that could start `text` (with and without a leading space)."""
        ids: list[int] = []
        for variant in (" " + text, text):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.append(int(toks[0]))
        seen: list[int] = []
        for i in ids:
            if i not in seen:
                seen.append(i)
        return seen

    @staticmethod
    def _bare_first(tokenizer, text: str) -> int:
        """First token of `text` without a leading space (collision semantics)."""
        toks = tokenizer.encode(text, add_special_tokens=False)
        return int(toks[0]) if toks else -1

    @classmethod
    def build(cls, f: Field, tokenizer) -> "CompiledField":
        if f.is_boolean:
            candidates = [cls._first_tokens(tokenizer, "true"), cls._first_tokens(tokenizer, "false")]
            sequences = [
                [int(t) for t in tokenizer.encode("true", add_special_tokens=False)],
                [int(t) for t in tokenizer.encode("false", add_special_tokens=False)],
            ]
            bare_firsts = [cls._bare_first(tokenizer, "true"), cls._bare_first(tokenizer, "false")]
            suffix = f'  "{f.name}": '
        else:
            prefix = os.path.commonprefix(f.choices)
            candidates = []
            sequences = []
            bare_firsts = []
            for choice in f.choices:
                remainder = choice[len(prefix):]
                toks = [int(t) for t in tokenizer.encode(remainder, add_special_tokens=False)] or [0]
                sequences.append(toks)
                candidates.append(cls._first_tokens(tokenizer, remainder))
                bare_firsts.append(cls._bare_first(tokenizer, remainder))
            suffix = f'  "{f.name}": "{prefix}'

        collision = len(set(bare_firsts)) < len(bare_firsts)
        return cls(
            field=f,
            suffix_tokens=[int(t) for t in tokenizer.encode(suffix, add_special_tokens=False)],
            candidate_ids=candidates,
            sequences=sequences,
            collision=collision,
        )

    def candidate_ids_for(self, answer_index: int) -> list[int]:
        return self.candidate_ids[answer_index]
