"""Schema-level tests. No model required — pure Python."""

import pytest

from parallel_decisions.schema import Field, Schema, SchemaError


def test_boolean_and_enum():
    schema = Schema({
        "flag": {"type": "boolean", "description": "a flag"},
        "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "risk level"},
    })
    assert len(schema) == 2
    assert schema["flag"].is_boolean
    assert schema["flag"].answers == ["true", "false"]
    assert schema["risk"].answers == ["LOW", "HIGH"]


def test_choice_mapping_carries_descriptions():
    schema = Schema({
        "risk": {"type": "enum", "choices": {"LOW": "small", "HIGH": "big"}, "description": "risk"},
    })
    field = schema["risk"]
    assert field.choices == ["LOW", "HIGH"]
    assert field.choice_descriptions == {"LOW": "small", "HIGH": "big"}


def test_rejects_bad_schemas():
    with pytest.raises(SchemaError):
        Schema({})
    with pytest.raises(SchemaError):
        Schema({"x": {"type": "number", "description": "nope"}})
    with pytest.raises(SchemaError):
        Schema({"x": {"type": "enum", "description": "no choices"}})
    with pytest.raises(SchemaError):
        Schema({"x": {"type": "enum", "choices": ["only_one"]}})
    with pytest.raises(SchemaError):
        Schema({"x": {"type": "enum", "choices": ["a", "a"]}})
    with pytest.raises(SchemaError):
        Schema({"x": {"type": "enum", "choices": [str(i) for i in range(300)]}})


def test_json_round_trip(tmp_path):
    original = Schema({
        "flag": {"type": "boolean", "description": "a flag"},
        "risk": {"type": "enum", "choices": {"LOW": "small", "HIGH": "big"}, "description": "risk"},
    })
    path = tmp_path / "schema.json"
    original.to_json(str(path))
    loaded = Schema.from_json(str(path))
    assert list(loaded.fields) == list(original.fields)
    assert loaded["risk"].choice_descriptions == {"LOW": "small", "HIGH": "big"}


def test_prompt_includes_descriptions_and_choices():
    from parallel_decisions.prompts import build_prompt

    schema = Schema({
        "risk": {"type": "enum", "choices": {"LOW": "small", "HIGH": "big"}, "description": "risk level"},
        "flag": {"type": "boolean", "description": "a flag"},
    })
    prompt = build_prompt("context text", schema)
    assert "risk level" in prompt
    assert "LOW" in prompt and "HIGH" in prompt
    assert "context text" in prompt
    assert prompt.rstrip().endswith("{")


class _StubTokenizer:
    """Maps whole strings to token ids; unknown strings fall back to a byte id."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0
        self._vocab: dict[str, int] = {}

    def _id(self, text: str) -> int:
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 1
        return self._vocab[text]

    def encode(self, text, add_special_tokens=False):
        # one token per word, a leading space attached to its word
        import re

        return [self._id(tok) for tok in re.findall(r" ?[^ ]+", text)]


def test_compilation_basic():
    tok = _StubTokenizer()
    schema = Schema({
        "plain": {"type": "enum", "choices": ["ALPHA", "BETA"], "description": "x"},
        "flag": {"type": "boolean", "description": "x"},
    })
    compiled = {c.field.name: c for c in schema.compile(tok)}
    assert compiled["plain"].collision is False
    assert compiled["flag"].collision is False
    assert all(c.suffix_tokens for c in compiled.values())
    # two candidates per enum field (the leading-space and bare variants collapse
    # into one entry per answer when the tokenizer maps them to distinct ids)
    assert len(compiled["plain"].candidate_ids) == 2


class _MapTokenizer:
    """Explicit string -> token-list map, for white-box collision tests."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        if text not in self.mapping:
            raise KeyError(f"unknown token string: {text!r}")
        return list(self.mapping[text])


def test_collision_detection_and_sequences():
    # choices share the common prefix "A"; the remainders "B" and "X" are given
    # deliberately colliding bare first tokens (both 1) to exercise the slow path.
    tok = _MapTokenizer({
        '  "f": "A': [90],
        " B": [5], "B": [1, 2],
        " X": [6], "X": [1, 3],
    })
    schema = Schema({"f": {"type": "enum", "choices": ["AB", "AX"], "description": "x"}})
    compiled = schema.compile(tok)[0]
    assert compiled.collision is True
    assert compiled.sequences == [[1, 2], [1, 3]]
    assert compiled.candidate_ids == [[5, 1], [6, 1]]
    assert compiled.suffix_tokens == [90]


def test_no_false_collision_on_shared_space_token():
    # real tokenizers render " 1" as [space, "1"]; the leading space is shared by
    # everything, so only the bare variants may decide whether a collision exists.
    tok = _MapTokenizer({
        '  "p": "P': [90],
        " 1": [220, 16], "1": [16],
        " 2": [220, 17], "2": [17],
    })
    schema = Schema({"p": {"type": "enum", "choices": ["P1", "P2"], "description": "x"}})
    compiled = schema.compile(tok)[0]
    assert compiled.collision is False
