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
    assert not prompt.rstrip().endswith("{{")


class _VocabTokenizer:
    """Greedy longest-match tokenizer over an explicit vocabulary.

    Real tokenizers pick pieces by BPE over a fixed vocabulary, so what a piece is
    does not depend on what came before it in the string. Longest-match reproduces
    that property, which is what makes it possible to test the interesting question:
    does the tokenizer absorb the suffix's trailing space into a suffix token (so the
    answer is emitted bare), or does the answer carry its own space?

    Unknown characters fall back to a per-character id, so nothing ever crashes.
    """

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.max_len = max((len(k) for k in self.vocab), default=1)

    def encode(self, text, add_special_tokens=False):
        out = []
        i = 0
        while i < len(text):
            for size in range(min(self.max_len, len(text) - i), 0, -1):
                piece = text[i:i + size]
                if piece in self.vocab:
                    out.append(self.vocab[piece])
                    i += size
                    break
            else:
                out.append(self.vocab.setdefault(text[i], 9000 + ord(text[i])))
                i += 1
        return out


# `": ` is a single piece, so the suffix eats the space and answers start bare.
# Deliberately excludes whole-answer tokens like "Alpha" so that multi-token
# answers can be exercised by supplying the pieces instead.
VOCAB = {
    '  "': 90, '"': 91, ": ": 93, '": ': 94,
    "true": 7, "false": 8,
    "A": 12, "B": 13, "X": 14,
    "extra": 50, "_approved": 51, "_unapproved": 52, "prorated": 53,
}
ALPHABET_VOCAB = {**VOCAB, "Al": 21, "pha": 22, "Be": 24, "ta": 25}


def _tok(**extra):
    return _VocabTokenizer({**VOCAB, **extra})


def test_compilation_basic():
    tok = _tok()
    schema = Schema({
        "plain": {"type": "enum", "choices": ["AB", "AX"], "description": "x"},
        "flag": {"type": "boolean", "description": "x"},
    })
    compiled = {c.field.name: c for c in schema.compile(tok)}
    assert compiled["plain"].collision is False
    assert compiled["flag"].collision is False
    assert all(c.suffix_tokens for c in compiled.values())
    # one candidate per allowed answer, read off the real encoding: "AB" / "AX"
    # share the prefix "A", so the decision is between "B" and "X"
    assert compiled["plain"].candidate_ids == [[13], [14]]
    assert compiled["flag"].candidate_ids == [[7], [8]]


def test_candidates_are_read_off_the_suffix_context():
    """The space after the colon belongs to the suffix, so answers start bare.

    Scoring ' true' alongside 'true' and keeping the higher logit shifts every
    probability in the field. The candidate must be the token the model actually
    emits next, which is what encoding `suffix + answer` tells us.
    """
    tok = _tok()
    compiled = Schema({"flag": {"type": "boolean", "description": "x"}}).compile(tok)[0]
    assert compiled.suffix == '  "flag": '
    assert compiled.suffix_tokens == tok.encode('  "flag": ')
    assert compiled.candidate_ids == [[7], [8]]
    assert compiled.sequences == [[7], [8]]


def test_sequences_are_the_true_continuation():
    tok = _VocabTokenizer(ALPHABET_VOCAB)
    schema = Schema({"f": {"type": "enum", "choices": ["Alpha", "Beta"], "description": "x"}})
    compiled = schema.compile(tok)[0]
    for choice, seq in zip(["Alpha", "Beta"], compiled.sequences):
        assert list(compiled.suffix_tokens) + list(seq) == tok.encode(compiled.suffix + choice)


def test_multitoken_answers_span_several_tokens():
    tok = _VocabTokenizer(ALPHABET_VOCAB)   # no whole-answer tokens available
    schema = Schema({"f": {"type": "enum", "choices": ["Alpha", "Beta"], "description": "x"}})
    compiled = schema.compile(tok)[0]
    assert compiled.candidate_ids == [[21], [24]]
    assert compiled.sequences == [[21, 22], [24, 25]]
    assert compiled.collision is False


def test_collision_when_answers_share_their_first_token():
    """A collision is exactly 'two answers emit the same next token'.

    This is the invoice case: `extra_approved` and `extra_unapproved` both continue
    with the `extra` token, so a first-token softmax cannot tell them apart and the
    full continuations have to be scored.
    """
    tok = _tok()
    schema = Schema({"f": {"type": "enum",
                           "choices": ["extra_approved", "extra_unapproved", "prorated"],
                           "description": "x"}})
    compiled = schema.compile(tok)[0]
    assert compiled.candidate_ids == [[50], [50], [53]]
    assert compiled.sequences == [[50, 51], [50, 52], [53]]
    assert compiled.collision is True


def test_no_collision_when_tokens_differ():
    tok = _tok()
    schema = Schema({"f": {"type": "enum", "choices": ["prorated", "extra_approved"],
                           "description": "x"}})
    compiled = schema.compile(tok)[0]
    assert compiled.candidate_ids == [[53], [50]]
    assert compiled.collision is False


def test_next_token_falls_back_when_the_boundary_merges():
    """If the tokenizer merges across the suffix boundary, use the answer's own tokens."""
    from parallel_decisions.schema import CompiledField

    class _Merging:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == '  "f": ':
                return [90]
            if text == '  "f": true':      # merged across the boundary
                return [91, 92]
            if text == "true":
                return [7]
            return [0]

    compiled = CompiledField.build(Field("f", "boolean", "x", ["true", "false"]), _Merging())
    assert compiled.suffix_tokens == [90]
    assert compiled.candidate_ids[0] == [7]     # the fallback path


def test_multi_field_is_accepted_and_round_trips():
    schema = Schema({
        "actions": {"type": "multi", "choices": ["retry", "escalate", "refund"],
                    "description": "which actions apply"},
    })
    field = schema["actions"]
    assert field.is_multi and not field.is_boolean
    assert field.choices == ["retry", "escalate", "refund"]
    assert field.answers == ["true", "false"]
    round_tripped = Schema.from_json(schema.to_json())
    assert round_tripped["actions"].is_multi
    assert round_tripped["actions"].choices == ["retry", "escalate", "refund"]


def test_multi_field_compiles_to_one_row_per_choice():
    tok = _tok(retry=40, escalate=41)
    schema = Schema({"actions": {"type": "multi", "choices": ["retry", "escalate"],
                                 "description": "x"}})
    compiled = schema.compile(tok)
    assert len(compiled) == 2
    assert [c.row_name for c in compiled] == ["actions[0]", "actions[1]"]
    assert [c.choice_index for c in compiled] == [0, 1]
    assert compiled[0].suffix == '  "actions[0]": '
    assert compiled[1].suffix == '  "actions[1]": '
    assert compiled[0].suffix_tokens != compiled[1].suffix_tokens
    assert all(c.field.name == "actions" for c in compiled)


def test_multi_field_rejects_choice_index_on_single_row():
    from parallel_decisions.schema import CompiledField

    with pytest.raises(SchemaError):
        CompiledField(field=Field("f", "boolean", "x", ["true", "false"]),
                      suffix='  "f": ', suffix_tokens=[90], candidate_ids=[[7], [8]],
                      sequences=[[7], [8]], collision=False, choice_index=0)


def test_multi_prompt_describes_indexed_keys():
    from parallel_decisions.prompts import build_prompt

    schema = Schema({
        "actions": {"type": "multi", "choices": {"retry": "try again", "refund": "give money back"},
                    "description": "which actions apply"},
    })
    prompt = build_prompt("ctx", schema)
    assert "multi-select" in prompt
    assert '"actions[i]"' in prompt
    assert "retry" in prompt and "try again" in prompt


def test_multi_assemble_uses_per_choice_decisions():
    from parallel_decisions.engine import Decider, FieldValue

    field = Field("actions", "multi", "x", ["a", "b", "c"])
    by_index = {
        0: FieldValue("actions", True, 0.9, [], distribution={"true": 0.9, "false": 0.1}),
        1: FieldValue("actions", False, 0.2, [], distribution={"true": 0.2, "false": 0.8}),
        2: FieldValue("actions", True, 0.55, [], distribution={"true": 0.55, "false": 0.45}),
    }
    fv = Decider._assemble_multi(field, by_index)
    assert fv.value == ["a", "c"]
    # the set's confidence is its weakest member
    assert fv.probability == pytest.approx(0.55)


def test_multi_assemble_empty_set_reports_confidence_in_emptiness():
    from parallel_decisions.engine import Decider, FieldValue

    field = Field("actions", "multi", "x", ["a", "b"])
    by_index = {
        0: FieldValue("actions", False, 0.3, [], distribution={"true": 0.3, "false": 0.7}),
        1: FieldValue("actions", False, 0.1, [], distribution={"true": 0.1, "false": 0.9}),
    }
    fv = Decider._assemble_multi(field, by_index)
    assert fv.value == []
    assert fv.probability == pytest.approx(0.7)  # confidence that nothing applies
