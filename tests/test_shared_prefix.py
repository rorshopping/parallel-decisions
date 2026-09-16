"""Shared schema prefix: reuse one prefill across many contexts.

The equivalence test needs the real tokenizer (the whole point is whether the
tokenizer keeps the prompt prefix intact), so it is skipped when `mlx_lm` is absent.
Everything else here is pure logic.
"""

from __future__ import annotations

import pytest

from parallel_decisions import Schema
from parallel_decisions.engine import PromptPrefix, _copy_cache
from parallel_decisions.prompts import build_prompt, build_prompt_parts

SCHEMA = Schema({
    "risk": {"type": "enum", "choices": ["low", "high"], "description": "risk level"},
    "flag": {"type": "boolean", "description": "a flag"},
})


def test_parts_reconstruct_the_prompt_exactly():
    """The split must be a pure refactor: no prompt text may change."""
    for context in ("short", "line\nwith\nnewlines", "unicode ✓ context"):
        prefix, remainder = build_prompt_parts(context, SCHEMA)
        assert prefix + remainder == build_prompt(context, SCHEMA)


def test_prefix_is_independent_of_the_context():
    a, _ = build_prompt_parts("one", SCHEMA)
    b, _ = build_prompt_parts("two", SCHEMA)
    assert a == b
    assert a.endswith("<|im_start|>user\n")


def test_prefix_contains_every_field_and_choice():
    prefix, _ = build_prompt_parts("ctx", SCHEMA)
    for token in ("risk", "flag", "low", "high"):
        assert token in prefix


def test_prefix_ends_before_the_context():
    prefix, remainder = build_prompt_parts("MY-CONTEXT", SCHEMA)
    assert not prefix.endswith("MY-CONTEXT")
    assert remainder.startswith("MY-CONTEXT")
    assert remainder.endswith("{\n")


def test_release_marks_the_prefix_unusable():
    prefix = PromptPrefix(decider=None, schema=SCHEMA, cache=["fake"],
                          prefix_tokens=[1, 2, 3], compiled=[])
    assert prefix.reusable
    assert prefix.prefix_tokens_count == 3
    prefix.release()
    assert not prefix.reusable


def test_copy_cache_is_shallow_and_preserves_arrays():
    class Layer:
        def __init__(self):
            self.keys = [1, 2, 3]
            self.values = [4, 5, 6]

    original = [Layer(), Layer()]
    copied = _copy_cache(original)
    assert len(copied) == 2
    assert copied is not original
    assert copied[0] is not original[0]
    assert copied[0].keys is original[0].keys      # arrays are shared, not copied


def test_decide_with_prefix_rejects_a_foreign_prefix():
    from parallel_decisions import Decider

    a = Decider(model_id="unused", warmup=False)
    b = Decider(model_id="unused", warmup=False)
    prefix = PromptPrefix(decider=a, schema=SCHEMA, cache=None,
                          prefix_tokens=[], compiled=[])
    with pytest.raises(ValueError, match="different Decider"):
        b.decide_with_prefix(prefix, "ctx")


def test_decide_with_prefix_rejects_empty_context():
    from parallel_decisions import Decider

    d = Decider(model_id="unused", warmup=False)
    prefix = PromptPrefix(decider=d, schema=SCHEMA, cache=None,
                          prefix_tokens=[], compiled=[])
    with pytest.raises(ValueError, match="non-empty"):
        d.decide_with_prefix(prefix, "   ")


@pytest.mark.parametrize("context", ["plain text", "starts with space", "日本語"])
def test_prefix_tokens_are_a_token_prefix_of_the_whole_prompt(context):
    """The reuse is only valid when the tokenizer does not merge across the boundary.

    This is the check `Decider.prepare()` performs before it commits to the shortcut;
    if a tokenizer ever breaks it, prepare() falls back to full prefill instead.
    """
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_tokenizer

    try:
        # the default model's tokenizer: the one the shared prefix must work with
        tokenizer = load_tokenizer("mlx-community/Qwen2.5-7B-Instruct-4bit")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"tokenizer unavailable: {exc}")

    prefix_text, _ = build_prompt_parts(context, SCHEMA)
    prefix_tokens = tokenizer.encode(prefix_text)
    whole_tokens = tokenizer.encode(build_prompt(context, SCHEMA))
    assert list(whole_tokens[:len(prefix_tokens)]) == list(prefix_tokens)
