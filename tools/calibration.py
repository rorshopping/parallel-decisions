#!/usr/bin/env python3
"""Calibration analysis: are the reported probabilities meaningful?

Run from the research tree (`jev-on-a-laptop` / `rlcd-research`), which holds the
full TypeSafe evaluation data and the recorded model answers. Kept here so the
CALIBRATION.md numbers can be reproduced.

For every answered (case, question) pair from the full TypeSafe eval, compare the
reported probability of the chosen answer against whether that answer matched the
reference. A calibrated model would show: of the pairs answered with ~0.9, about
90% are correct.

Output: reliability table (buckets), ECE (expected calibration error), and the
confidence distribution for correct vs incorrect answers.

Usage:
  .venv/bin/python evals/calibration.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "full_eval.json")
RESULTS = os.path.join(HERE, "results")

MODELS = {
    "local-qwen2.5-7b": "full-local-qwen2.5-7b.json",
    "local-qwen3-8b": "full-local-qwen3-8b.json",
}
BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def consensus(ref: dict):
    probs = ref.get("probs") or {}
    if probs:
        return str(max(probs, key=probs.get))
    value = ref.get("value")
    return str(value) if value is not None else None


def canonical(value, qtype: str) -> str | None:
    if value is None:
        return None
    if qtype == "noul":
        try:
            return str(float(value) >= 0.5).lower()
        except (TypeError, ValueError):
            return None
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def main() -> None:
    full = json.load(open(FULL))
    qtype = {}
    ref_by = {}
    for wf, wdata in full["workflows"].items():
        for case in wdata["cases"]:
            ref_by[(wf, case["case_id"])] = case["reference"]
            for q in case["questions"]:
                qtype[(wf, q["qid"])] = q["type"]

    for model, path in MODELS.items():
        p = os.path.join(RESULTS, path)
        if not os.path.isfile(p):
            print(f"(skip {model})")
            continue
        data = json.load(open(p))["answers"]

        buckets = defaultdict(lambda: [0, 0])   # bucket -> [correct, total]
        conf_correct: list[float] = []
        conf_wrong: list[float] = []
        n = 0
        for wf, cases in data.items():
            for cid, answers in cases.items():
                ref = ref_by.get((wf, cid), {})
                for qid, ans in answers.items():
                    if qid not in ref:
                        continue
                    want = consensus(ref[qid])
                    if want is None:
                        continue
                    t = qtype.get((wf, qid))
                    if t is None:
                        continue
                    got = canonical(ans.get("raw"), t)
                    if got is None:
                        continue
                    # probability attached to the chosen answer
                    prob = ans.get("prob")
                    if prob is None:
                        prob = float(ans["raw"]) if t == "noul" and isinstance(ans.get("raw"), (int, float)) else None
                    conf = None
                    if t == "noul" and isinstance(prob, (int, float)):
                        raw = float(ans["raw"])
                        conf = raw if got == "true" else 1.0 - raw
                    if conf is None:
                        continue
                    correct = got == want
                    n += 1
                    for lo, hi in BUCKETS:
                        if lo <= conf < hi:
                            buckets[(lo, hi)][1] += 1
                            buckets[(lo, hi)][0] += correct
                            break
                    (conf_correct if correct else conf_wrong).append(conf)

        print(f"\n===== {model} — {n} scored pairs =====")
        print(f"{'confidence':<14} {'n':>5} {'accuracy':>9} {'gap':>7}")
        ece = 0.0
        for lo, hi in BUCKETS:
            ok, total = buckets.get((lo, hi), [0, 0])
            if total == 0:
                continue
            acc = ok / total
            mid = (lo + min(hi, 1.0)) / 2
            gap = acc - mid
            ece += abs(gap) * total / max(n, 1)
            print(f"{lo:.1f}-{hi if hi <= 1 else 1.0:.1f}      {total:>5} {acc:>8.1%} {gap:>+7.1%}")
        print(f"ECE (expected calibration error): {ece:.3f}")
        if conf_correct and conf_wrong:
            print(f"mean confidence | correct  : {sum(conf_correct)/len(conf_correct):.3f} (n={len(conf_correct)})")
            print(f"mean confidence | incorrect: {sum(conf_wrong)/len(conf_wrong):.3f} (n={len(conf_wrong)})")
            hi_conf_wrong = sum(1 for c in conf_wrong if c >= 0.9)
            print(f"incorrect answers given >=0.90 confidence: {hi_conf_wrong}/{len(conf_wrong)}")


if __name__ == "__main__":
    main()
