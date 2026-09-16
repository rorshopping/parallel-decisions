#!/usr/bin/env python3
"""Fit a calibrator from labelled decisions — the roadmap's `tools/fit_temperature.py`.

The implementation lives in the library (`parallel_decisions.calibration`), so it is
usable from Python, and the same thing is available as `pd calibrate`. This script is
a thin CLI so the file exists where the roadmap points, and so it can be run straight
from a checkout without installing the console script.

Usage:
    python tools/fit_calibration.py --data evals/results/calibration/qwen2.5-7b.jsonl \
        --out calibration.json
    python tools/fit_calibration.py --data labels.jsonl --method temperature --json

Input: one JSON object per line (or a JSON list), each with
    distribution: {label: probability, ...}    (or "probability" for binary data)
and either
    correct: true/false                        (was the top choice right?)
    target:  "label"                           (the right label)

Output: the calibrated comparison table, plus a calibrator JSON with --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parallel_decisions.cli import main as pd_main  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args = ["calibrate"] + args
    return pd_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
