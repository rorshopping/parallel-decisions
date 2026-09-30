"""Equivalence tests for `calibration.auroc`.

The shipped implementation uses the Mann-Whitney / rank-sum form. These tests
pin it against the previous O(P x N) pairwise double loop (kept here verbatim
as the oracle) and against hand-computed values.

TOLERANCE
---------
The pairwise form accumulates `wins` one pair at a time; the rank form computes
`U = R_pos - n_pos(n_pos+1)/2` and divides once. Different summation orders over
the same rational quantity legitimately differ in the last ULP, so equality is
asserted to a tolerance, not bit-exactly:

    TOL = 1e-9

1e-9 is ~7 orders of magnitude tighter than the 0.65 / "is this metric
informative" comparisons the value feeds, and far tighter than the ~1e-12
relative error actually observed (worst case below is 0.0, i.e. these cases
agree exactly). Where a case is exactly representable the test additionally
asserts `==` to prove the two forms are not merely close.
"""
from __future__ import annotations

import math
import random

import pytest

from parallel_decisions.calibration import auroc

TOL = 1e-9


def auroc_old(confs, corrects):
    """Verbatim copy of the previous O(P x N) implementation (calibration.py:379-393)."""
    pos = [float(c) for c, ok in zip(confs, corrects) if ok]
    neg = [float(c) for c, ok in zip(confs, corrects) if not ok]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def _check(confs, corrects, *, exact=False):
    got = auroc(confs, corrects)
    want = auroc_old(confs, corrects)
    if math.isnan(want):
        assert math.isnan(got), f"expected NaN, got {got!r}"
        return
    assert got == pytest.approx(want, abs=TOL), f"new={got!r} old={want!r}"
    if exact:
        assert got == want, f"expected bit-exact {want!r}, got {got!r}"
    return got


# --------------------------------------------------------------- named shapes

def test_perfect_separation_is_one():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    _check([0.9, 0.8, 0.7, 0.2, 0.1], [True, True, True, False, False], exact=True)


def test_inverted_is_zero():
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    _check([0.2, 0.1, 0.9, 0.8], [False, False, True, True], exact=True)


def test_heavy_ties_all_identical_is_one_half():
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5
    # 2 pos and 3 neg all at the same score: every one of the 6 pairs is a tie
    assert auroc([0.7] * 5, [True, True, False, False, False]) == 0.5


def test_single_class_edges_return_nan():
    assert math.isnan(auroc([0.9, 0.8], [True, True]))        # no negatives
    assert math.isnan(auroc([0.2, 0.1], [False, False]))      # no positives
    assert math.isnan(auroc([0.5], [True]))
    assert math.isnan(auroc([0.5], [False]))


def test_empty_input_returns_nan():
    assert math.isnan(auroc([], []))
    assert math.isnan(auroc((), ()))


def test_single_pair():
    # one positive, one negative -> exactly one compared pair
    assert auroc([0.9, 0.1], [True, False]) == 1.0
    assert auroc([0.1, 0.9], [True, False]) == 0.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    # a lone sample is single-class either way
    assert math.isnan(auroc([0.5], [True]))
    assert math.isnan(auroc([0.5], [False]))


# ------------------------------------------------------------------- ties

def test_partial_ties_give_half_credit():
    # pos = {0.9, 0.5}, neg = {0.5, 0.1}
    # pairs: (0.9,0.5)=win  (0.9,0.1)=win  (0.5,0.5)=tie  (0.5,0.1)=win
    # wins = 1 + 1 + 0.5 + 1 = 3.5 / 4 = 0.875
    assert auroc([0.9, 0.5, 0.5, 0.1], [True, True, False, False]) == 0.875
    _check([0.9, 0.5, 0.5, 0.1], [True, True, False, False], exact=True)


def test_ties_within_the_positive_class_do_not_matter():
    base = [0.9, 0.8, 0.3, 0.2]
    labels = [True, True, False, False]
    _check(base, labels, exact=True)
    # duplicating a positive on an identical score leaves the statistic alone
    _check([0.9, 0.8, 0.8, 0.3, 0.2], [True, True, True, False, False])


def test_average_rank_tie_handling_matches_hand_computation():
    # 3 pos, 3 neg, scores: pos={0.1,0.5,0.9}, neg={0.1,0.5,0.9}
    # every pos ties with exactly one neg and beats/loses the others:
    #   pos 0.1: vs 0.1 tie (0.5), vs 0.5 lose, vs 0.9 lose  -> 0.5
    #   pos 0.5: vs 0.1 win,   vs 0.5 tie (0.5), vs 0.9 lose  -> 1.5
    #   pos 0.9: vs 0.1 win,   vs 0.5 win,  vs 0.9 tie (0.5)-> 2.5
    # wins = 4.5 / 9 = 0.5
    assert auroc([0.1, 0.5, 0.9, 0.1, 0.5, 0.9], [True, True, True, False, False, False]) == 0.5
    _check([0.1, 0.5, 0.9, 0.1, 0.5, 0.9], [True, True, True, False, False, False], exact=True)


def test_large_heavy_tie_block():
    # 40 positives and 40 negatives drawn from only 5 distinct scores -> lots of ties
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    confs, corrects = [], []
    for i in range(40):
        confs.append(scores[i % 5])
        corrects.append(True)
    for i in range(40):
        confs.append(scores[(i + 2) % 5])
        corrects.append(False)
    _check(confs, corrects)


def test_signed_zero_and_negative_zero_tie():
    # -0.0 == 0.0 in Python, so these are ties, not wins
    _check([0.0, -0.0, 1.0], [True, False, False])
    assert auroc([0.0, -0.0], [True, False]) == 0.5


# ------------------------------------------------- exhaustive small corpora

def test_exhaustive_over_all_labelings_of_four_scores():
    scores = [0.1, 0.5, 0.9]
    n = 0
    for mask in range(1 << len(scores)):
        corrects = [bool(mask >> i & 1) for i in range(len(scores))]
        _check(scores, corrects)
        n += 1
    assert n == 8


def test_exhaustive_over_all_multisets_of_three_scores():
    # every non-decreasing triple from {0,1,2,3} x every labeling
    for a in range(4):
        for b in range(a, 4):
            for c in range(b, 4):
                scores = [a / 3.0, b / 3.0, c / 3.0]
                for mask in range(8):
                    corrects = [bool(mask >> i & 1) for i in range(3)]
                    _check(scores, corrects)


# ------------------------------------------------------------- randomised

@pytest.mark.parametrize("n_pos,n_neg,seed", [
    (1, 1, 1), (1, 40, 2), (40, 1, 3), (7, 13, 4), (50, 50, 5),
    (500, 300, 6), (1000, 1000, 7), (200, 2000, 8),
])
def test_matches_old_on_random_continuous_scores(n_pos, n_neg, seed):
    rng = random.Random(seed)
    confs, corrects = [], []
    for _ in range(n_pos):
        confs.append(rng.random())
        corrects.append(True)
    for _ in range(n_neg):
        confs.append(rng.random())
        corrects.append(False)
    rng.shuffle(confs), rng.shuffle(corrects)
    # shuffle labels independently of scores so the signal is ~0.5
    pairs = list(zip(confs, corrects))
    rng.shuffle(pairs)
    _check([c for c, _ in pairs], [k for _, k in pairs])


@pytest.mark.parametrize("n_pos,n_neg,seed", [
    (100, 100, 11), (300, 300, 12), (500, 200, 13),
])
def test_matches_old_on_random_low_cardinality_scores(n_pos, n_neg, seed):
    """Scores quantised to 4 levels: ties everywhere, the classic regression case."""
    rng = random.Random(seed)
    levels = [0.0, 1 / 3, 2 / 3, 1.0]
    pairs = [(rng.choice(levels), True) for _ in range(n_pos)]
    pairs += [(rng.choice(levels), False) for _ in range(n_neg)]
    rng.shuffle(pairs)
    _check([c for c, _ in pairs], [k for _, k in pairs])


def test_matches_old_on_a_large_random_case():
    """Realistically sized evaluation set: 5,000 x 5,000 (25M pairwise comparisons)."""
    rng = random.Random(20260926)
    pairs = [(rng.betavariate(6, 2), True) for _ in range(5000)]
    pairs += [(rng.betavariate(2, 6), False) for _ in range(5000)]
    rng.shuffle(pairs)
    confs = [c for c, _ in pairs]
    corrects = [k for _, k in pairs]
    got = _check(confs, corrects)
    assert 0.9 < got < 1.0  # sanity: the signal really is there


def test_order_of_inputs_does_not_matter():
    rng = random.Random(99)
    pairs = [(rng.random(), bool(rng.getrandbits(1))) for _ in range(400)]
    base = auroc([c for c, _ in pairs], [k for _, k in pairs])
    for _ in range(5):
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        assert auroc([c for c, _ in shuffled], [k for _, k in shuffled]) == base
    _check([c for c, _ in pairs], [k for _, k in pairs])


def test_result_is_always_within_zero_and_one():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randrange(2, 40)
        pairs = [(rng.choice([0.0, 0.5, 1.0, rng.random()]), bool(rng.getrandbits(1)))
                 for _ in range(n)]
        value = auroc([c for c, _ in pairs], [k for _, k in pairs])
        if not math.isnan(value):
            assert 0.0 <= value <= 1.0


def test_extra_unpaired_inputs_are_ignored_like_before():
    # zip() truncates, so a longer `corrects` is silently dropped
    _check([0.9, 0.1, 0.5], [True, False, True, False, True])
