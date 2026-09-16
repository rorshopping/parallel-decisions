"""Calibration: make `probability` something you can threshold on.

The engine reports `softmax(logits)` over a field's allowed answers. That number
ranks answers inside a field well, but it is not the probability that the answer
is *correct* — measured ECE on the public TypeSafe eval was ~0.09, with almost
everything piling into the top bucket (see `CALIBRATION.md`).

This module fixes the scale, not the decisions:

    from parallel_decisions import Calibrator
    cal = Calibrator.fit(records, method="temperature")   # or "platt"/"isotonic"
    cal.transform({"true": 0.97, "false": 0.03})          # -> {"true": 0.88, ...}

Three standard post-hoc methods are supported, all fitted on labelled data and all
**argument-order preserving** (the top choice never changes, so accuracy is
untouched and only the reported confidence moves):

- ``temperature``  one scalar T: ``p' ∝ p ** (1/T)`` (equivalent to softmax(z/T))
- ``platt``        logistic map of the top confidence: ``σ(a·logit(c) + b)``
- ``isotonic``     monotone step fit of top confidence to empirical accuracy

Because all three are monotone in the top confidence, they cannot change the
*order* of answers — they only change which threshold buys which error rate. Use
`risk_coverage()` to see that trade-off before picking a routing policy.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

EPS = 1e-9
DEFAULT_BINS = 10
METHODS = ("temperature", "platt", "isotonic")

__all__ = [
    "CalibrationError",
    "filter_records",
    "slices",
    "CalibrationRecord",
    "Calibrator",
    "CalibrationFit",
    "METRICS",
    "METHODS",
    "accuracy",
    "adaptive_ece",
    "auroc",
    "brier",
    "clip",
    "confidence",
    "ece",
    "fit_calibration",
    "load_records",
    "logit",
    "normalize",
    "reliability_table",
    "risk_coverage",
    "sigmoid",
    "softmax",
    "top_nll",
    "wilson_interval",
]


class CalibrationError(ValueError):
    """Raised for malformed calibration data or parameters."""


# --------------------------------------------------------------------- maths
def clip(p: float) -> float:
    """Clamp a probability away from 0/1 so logit() stays finite."""
    return min(1.0 - EPS, max(EPS, float(p)))


def logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(scores: Sequence[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def normalize(dist: Mapping[str, float]) -> dict[str, float]:
    """Non-negative weights -> probabilities that sum to 1."""
    clean = {str(k): max(0.0, float(v)) for k, v in dist.items()}
    total = sum(clean.values())
    if total <= 0.0:
        n = len(clean) or 1
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def confidence(dist: Mapping[str, float]) -> float:
    """Probability of the top choice."""
    if not dist:
        raise CalibrationError("empty distribution")
    return max(float(v) for v in dist.values())


def _argmax(dist: Mapping[str, float]) -> str:
    return max(dist, key=lambda k: (float(dist[k]), k))


# --------------------------------------------------------------- the records
@dataclass
class CalibrationRecord:
    """One labelled decision: a probability distribution and what was right.

    `correct` is whether the top choice matched the target. `target` is optional
    (it is needed only if you want exact full-distribution NLL).
    """

    distribution: dict[str, float]
    correct: bool
    chosen: str = ""
    target: str | None = None
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.distribution = normalize(self.distribution)
        if not self.chosen:
            self.chosen = _argmax(self.distribution)
        self.correct = bool(self.correct)

    @property
    def confidence(self) -> float:
        return confidence(self.distribution)

    @property
    def correct_confidence(self) -> float:
        """Probability mass on the correct label (needs `target`)."""
        if self.target is None:
            raise CalibrationError("record has no target")
        return float(self.distribution.get(self.target, 0.0))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationRecord":
        """Accept the shapes the evals and hand-labelled files actually use.

        - ``{"distribution": {...}, "target": "yes"}``
        - ``{"distribution": {...}, "correct": true}``
        - ``{"probability": 0.93, "correct": false}``      (binary, label-free)
        """
        if not isinstance(data, Mapping):
            raise CalibrationError(f"record must be an object, got {type(data).__name__}")
        if "distribution" in data and isinstance(data["distribution"], Mapping):
            dist = {str(k): float(v) for k, v in data["distribution"].items()}
        elif "probability" in data:
            p = clip(float(data["probability"]))
            if not 0.0 <= float(data["probability"]) <= 1.0:
                raise CalibrationError(f"probability out of range: {data['probability']}")
            dist = {"true": p, "false": 1.0 - p}
        else:
            raise CalibrationError("record needs a 'distribution' or a 'probability'")
        if len(dist) < 2:
            raise CalibrationError("distribution needs at least two labels")

        target = data.get("target")
        target = str(target) if target is not None else None
        chosen = data.get("chosen")
        chosen = str(chosen) if chosen is not None else ""

        if data.get("correct") is not None:
            correct = bool(data["correct"])
        elif target is not None:
            top = chosen or _argmax(normalize(dist))
            correct = top == target
        else:
            raise CalibrationError("record needs 'correct' or 'target'")

        meta = {k: data[k] for k in ("workflow", "case_id", "qid", "type", "model") if k in data}
        return cls(distribution=normalize(dist), correct=correct, chosen=chosen,
                   target=target, weight=float(data.get("weight", 1.0)), meta=meta)


def load_records(data: str | os.PathLike[str] | Iterable[Mapping[str, Any]]) -> list[CalibrationRecord]:
    """Load records from a .jsonl file, a .json file/list, or an iterable of dicts."""
    items: list[Mapping[str, Any]]
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        if not os.path.isfile(path):
            raise CalibrationError(f"no such file: {path}")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith(".jsonl") or (text.lstrip() and text.lstrip()[0] == "{" and "\n{" in text):
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            items = parsed["records"] if isinstance(parsed, Mapping) and "records" in parsed else parsed
    else:
        items = list(data)
    records = [CalibrationRecord.from_dict(item) for item in items]
    if not records:
        raise CalibrationError("no records loaded")
    return records


def filter_records(records: Sequence[CalibrationRecord],
                   *conditions: str) -> list[CalibrationRecord]:
    """Keep records matching every `key=value` condition.

    Keys are matched against `meta` (which carries `type`, `qid`, `workflow`, `model`
    when the source file has them), then against the record's own fields. Values
    compare as strings, with `true`/`false` understood for booleans::

        filter_records(records, "type=noul")        # fit the booleans only
        filter_records(records, "workflow=invoices")
    """
    out = list(records)
    for condition in conditions:
        if "=" not in condition:
            raise CalibrationError(f"condition must be key=value, got {condition!r}")
        key, _, raw = condition.partition("=")
        key, raw = key.strip(), raw.strip()
        wanted: Any = raw
        if raw.lower() in ("true", "false"):
            wanted = raw.lower() == "true"
        kept = []
        for record in out:
            value = record.meta.get(key, getattr(record, key, None))
            if value is None:
                continue
            if isinstance(value, bool) or isinstance(wanted, bool):
                match = bool(value) == bool(wanted)
            else:
                match = str(value) == raw
            if match:
                kept.append(record)
        out = kept
    if not out:
        raise CalibrationError(f"no records match {' '.join(conditions)}")
    return out


def slices(records: Sequence[CalibrationRecord], key: str = "type") -> dict[str, list[CalibrationRecord]]:
    """Group records by a meta key, for per-slice calibration and reporting."""
    grouped: dict[str, list[CalibrationRecord]] = {}
    for record in records:
        value = record.meta.get(key)
        if value is None:
            continue
        grouped.setdefault(str(value), []).append(record)
    return grouped


# -------------------------------------------------------------------- metrics
def ece(confs: Sequence[float], corrects: Sequence[bool], bins: int = DEFAULT_BINS) -> float:
    """Expected calibration error over equal-*width* confidence bins."""
    return _bin_gap(confs, corrects, bins, adaptive=False)


def adaptive_ece(confs: Sequence[float], corrects: Sequence[bool],
                 bins: int = DEFAULT_BINS) -> float:
    """ECE over equal-*count* bins — robust when confidence is heavily skewed.

    The engine puts ~90% of answers in the top bucket, where equal-width ECE is
    decided by a single bin. Equal-count bins give the tail of the distribution a
    vote too, which is what makes this the better model-selection signal here.
    """
    return _bin_gap(confs, corrects, bins, adaptive=True)


def _bin_gap(confs: Sequence[float], corrects: Sequence[bool], bins: int,
             adaptive: bool) -> float:
    rows = reliability_table(confs, corrects, bins=bins, adaptive=adaptive)
    n = sum(r["n"] for r in rows)
    if not n:
        return 0.0
    return sum(abs(r["accuracy"] - r["confidence"]) * r["n"] for r in rows) / n


def reliability_table(confs: Sequence[float], corrects: Sequence[bool], bins: int = DEFAULT_BINS,
                      adaptive: bool = False) -> list[dict[str, float]]:
    """Per-bin [n, mean confidence, empirical accuracy] rows, skipping empty bins.

    With ``adaptive=True`` the bin edges are empirical quantiles of the confidence
    values, so records that share a confidence always land in the same bin. This
    matters here: a heavy cluster of answers at the same rounded score would
    otherwise be split by input order, which made the number depend on shuffling.
    """
    pairs = sorted(zip(confs, corrects), key=lambda item: float(item[0]))
    if not pairs:
        return []
    if adaptive:
        edges = _quantile_edges([float(c) for c, _ in pairs], bins)
        groups: list[list[tuple[float, bool]]] = [[] for _ in range(len(edges) + 1)]
        for conf, ok in pairs:
            groups[_bisect_left(edges, float(conf))].append((float(conf), bool(ok)))
        groups = [g for g in groups if g]
    else:
        width = 1.0 / bins
        groups = [[] for _ in range(bins)]
        for conf, ok in pairs:
            idx = min(bins - 1, int(float(conf) / width))
            groups[idx].append((float(conf), bool(ok)))
        groups = [g for g in groups if g]

    rows = []
    for group in groups:
        confs = [c for c, _ in group]
        acc = sum(1 for _, ok in group if ok) / len(group)
        rows.append({
            "n": len(group),
            "confidence": sum(confs) / len(confs),
            "accuracy": acc,
            "low": min(confs),
            "high": max(confs),
        })
    return rows


def _quantile_edges(sorted_values: Sequence[float], bins: int) -> list[float]:
    """`bins - 1` interior quantile edges of an already-sorted sample."""
    n = len(sorted_values)
    edges: list[float] = []
    for b in range(1, bins):
        pos = b * n / bins
        lo = min(n - 1, max(0, int(math.floor(pos))))
        hi = min(n - 1, lo + 1)
        frac = pos - lo
        edges.append(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))
    return edges


def _bisect_left(edges: Sequence[float], value: float) -> int:
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def top_nll(confs: Sequence[float], corrects: Sequence[bool]) -> float:
    """Negative log-likelihood of the *top choice* being correct.

    For a binary field this is the exact NLL of the chosen label. For multi-class
    fields it is the honest "did the decision come out right" likelihood, which is
    the quantity a routing policy actually cares about.
    """
    if not confs:
        return 0.0
    total = 0.0
    for conf, ok in zip(confs, corrects):
        total += -math.log(clip(conf)) if ok else -math.log(clip(1.0 - conf))
    return total / len(confs)


def brier(confs: Sequence[float], corrects: Sequence[bool]) -> float:
    if not confs:
        return 0.0
    return sum((float(c) - float(ok)) ** 2 for c, ok in zip(confs, corrects)) / len(confs)


def accuracy(confs: Sequence[float], corrects: Sequence[bool]) -> float:
    if not confs:
        return 0.0
    return sum(1.0 for ok in corrects if ok) / len(corrects)


def auroc(confs: Sequence[float], corrects: Sequence[bool]) -> float:
    """P(confidence of a correct answer > confidence of a wrong one).

    0.5 means confidence carries no information about correctness, and no amount
    of post-hoc calibration will make it a useful routing signal.
    """
    pos = [float(c) for c, ok in zip(confs, corrects) if ok]
    neg = [float(c) for c, ok in zip(confs, corrects) if not ok]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def risk_coverage(confs: Sequence[float], corrects: Sequence[bool],
                  coverages: Sequence[float] | None = None) -> list[dict[str, float]]:
    """Error rate of the most-confident `coverage` share of answers.

    This is the routing table: at coverage 0.5 you act on the top half of answers
    by confidence, and the risk column is the error rate you inherit.
    """
    pairs = sorted(zip(confs, corrects), key=lambda item: -float(item[0]))
    n = len(pairs)
    if not n:
        return []
    coverages = list(coverages) if coverages is not None else [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows = []
    for cov in coverages:
        k = max(1, min(n, int(round(cov * n))))
        head = pairs[:k]
        errors = sum(1 for _, ok in head if not ok)
        rows.append({
            "coverage": k / n,
            "taken": k,
            "threshold": float(head[-1][0]),
            "errors": errors,
            "risk": errors / k,
            "risk_upper95": wilson_interval(errors, k)[1],
        })
    return rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate — honest error bars for routing."""
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


METRICS = ("ece", "ece_adaptive", "top_nll", "brier", "auroc", "accuracy")


def evaluate(confs: Sequence[float], corrects: Sequence[bool], bins: int = DEFAULT_BINS) -> dict[str, float]:
    """All metrics for one set of confidences."""
    return {
        "n": len(confs),
        "accuracy": accuracy(confs, corrects),
        "ece": ece(confs, corrects, bins=bins),
        "ece_adaptive": adaptive_ece(confs, corrects, bins=bins),
        "top_nll": top_nll(confs, corrects),
        "brier": brier(confs, corrects),
        "auroc": auroc(confs, corrects),
        "mean_confidence": mean(confs) if confs else 0.0,
    }


# ------------------------------------------------------------------ calibrator
@dataclass
class Calibrator:
    """A fitted map from raw softmax confidence to a calibrated probability."""

    kind: str = "identity"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    isotonic_x: list[float] = field(default_factory=list)
    isotonic_y: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in METHODS + ("identity",):
            raise CalibrationError(f"unknown calibration kind: {self.kind!r}")
        if self.kind == "temperature" and self.temperature <= 0:
            raise CalibrationError("temperature must be positive")

    # -- the transform -------------------------------------------------------
    def transform_confidence(self, conf: float) -> float:
        """Calibrated probability for a raw top-choice confidence."""
        conf = clip(conf)
        if self.kind == "identity":
            return conf
        if self.kind == "temperature":
            # binary view of the distribution: p^(1/T) / (p^(1/T) + (1-p)^(1/T))
            a = conf ** (1.0 / self.temperature)
            b = (1.0 - conf) ** (1.0 / self.temperature)
            return a / (a + b) if (a + b) > 0 else conf
        if self.kind == "platt":
            return sigmoid(self.platt_a * logit(conf) + self.platt_b)
        if self.kind == "isotonic":
            return self._isotonic(conf)
        raise CalibrationError(f"unknown calibration kind: {self.kind!r}")

    def _isotonic(self, conf: float) -> float:
        xs, ys = self.isotonic_x, self.isotonic_y
        if not xs:
            return clip(conf)
        if len(xs) == 1:
            return clip(ys[0])
        if conf <= xs[0]:
            return clip(ys[0])
        if conf >= xs[-1]:
            return clip(ys[-1])
        for i in range(1, len(xs)):
            if conf <= xs[i]:
                span = xs[i] - xs[i - 1]
                t = 0.0 if span <= 0 else (conf - xs[i - 1]) / span
                return clip(ys[i - 1] + t * (ys[i] - ys[i - 1]))
        return clip(ys[-1])

    def transform(self, dist: Mapping[str, float]) -> dict[str, float]:
        """Recalibrate a whole distribution, keeping its shape below the top."""
        probs = normalize(dist)
        if self.kind == "identity" or len(probs) < 2:
            return probs
        if self.kind == "temperature":
            # exact softmax(z / T): renormalise every class, not just the top
            return _temperature_transform(probs, self.temperature)
        return self._transform_top(probs)

    def _transform_top(self, probs: Mapping[str, float]) -> dict[str, float]:
        """Confidence-map based methods: move the top, scale the rest in proportion."""
        top = _argmax(probs)
        new_top = self.transform_confidence(float(probs[top]))
        rest_mass = 1.0 - new_top
        old_rest = 1.0 - float(probs[top])
        out = {}
        for label, value in probs.items():
            if label == top:
                out[label] = new_top
            elif old_rest <= EPS:
                out[label] = rest_mass / max(1, len(probs) - 1)
            else:
                out[label] = float(value) / old_rest * rest_mass
        return normalize(out)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_x": self.isotonic_x,
            "isotonic_y": self.isotonic_y,
            "meta": self.meta,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Calibrator":
        if not isinstance(data, Mapping):
            raise CalibrationError("calibrator must be a JSON object")
        return cls(
            kind=str(data.get("kind", "identity")),
            temperature=float(data.get("temperature", 1.0)),
            platt_a=float(data.get("platt_a", 1.0)),
            platt_b=float(data.get("platt_b", 0.0)),
            isotonic_x=[float(x) for x in data.get("isotonic_x", [])],
            isotonic_y=[float(y) for y in data.get("isotonic_y", [])],
            meta=dict(data.get("meta", {})),
        )

    @classmethod
    def from_json(cls, path_or_text: str) -> "Calibrator":
        if os.path.exists(path_or_text):
            with open(path_or_text, encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(path_or_text)
        if isinstance(data, Mapping) and "calibrator" in data:
            data = data["calibrator"]
        return cls.from_dict(data)

    # -- fitting -------------------------------------------------------------
    @classmethod
    def fit(cls, records: Sequence[CalibrationRecord], method: str = "temperature",
            **_: Any) -> "Calibrator":
        if not records:
            raise CalibrationError("cannot fit on zero records")
        if method == "temperature":
            return cls._fit_temperature(records)
        if method == "platt":
            return cls._fit_platt(records)
        if method == "isotonic":
            return cls._fit_isotonic(records)
        raise CalibrationError(f"unknown method: {method!r} (choose from {METHODS})")

    @classmethod
    def _fit_temperature(cls, records: Sequence[CalibrationRecord]) -> "Calibrator":
        """One scalar T minimising top-choice NLL over the *full* distributions."""
        def loss(log_t: float) -> float:
            t = math.exp(log_t)
            confs = [max(_temperature_transform(r.distribution, t).values()) for r in records]
            return top_nll(confs, [r.correct for r in records])

        lo, hi = math.log(0.05), math.log(50.0)
        for _ in range(200):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if loss(m1) < loss(m2):
                hi = m2
            else:
                lo = m1
        temperature = math.exp((lo + hi) / 2.0)
        confs_cal = [max(_temperature_transform(r.distribution, temperature).values())
                     for r in records]
        return cls(kind="temperature", temperature=round(temperature, 6),
                   meta={"n": len(records), "fit": "nll",
                         **evaluate(confs_cal, [r.correct for r in records])})

    @staticmethod
    def _temp_confidence(conf: float, temperature: float) -> float:
        """Binary view of temperature scaling (used when only a confidence is known)."""
        conf = clip(conf)
        a = conf ** (1.0 / temperature)
        b = (1.0 - conf) ** (1.0 / temperature)
        return a / (a + b) if (a + b) > 0 else conf

    @classmethod
    def _fit_platt(cls, records: Sequence[CalibrationRecord]) -> "Calibrator":
        """Logistic regression of correctness on logit(confidence), 2 params, IRLS."""
        xs = [logit(r.confidence) for r in records]
        ys = [1.0 if r.correct else 0.0 for r in records]
        a, b = 1.0, 0.0
        ridge = 1e-6
        for _ in range(100):
            g0 = g1 = h00 = h01 = h11 = 0.0
            for x, y in zip(xs, ys):
                p = sigmoid(a * x + b)
                w = p * (1.0 - p)
                g0 += (p - y) * x
                g1 += (p - y)
                h00 += w * x * x
                h01 += w * x
                h11 += w
            h00 += ridge
            h11 += ridge
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-18:
                break
            da = (h11 * g0 - h01 * g1) / det
            db = (h00 * g1 - h01 * g0) / det
            a -= da
            b -= db
            if abs(da) + abs(db) < 1e-12:
                break
        note = {}
        if a <= 0.0:  # confidence anti-correlated with correctness: refuse to flip
            a, b = 1.0, logit(max(1e-6, min(1 - 1e-6, mean(ys))))
            note["platt_clamped"] = True
        confs = [sigmoid(a * logit(r.confidence) + b) for r in records]
        return cls(kind="platt", platt_a=round(a, 6), platt_b=round(b, 6),
                   meta={"n": len(records), "fit": "nll", **note,
                         **evaluate(confs, [r.correct for r in records])})

    @classmethod
    def _fit_isotonic(cls, records: Sequence[CalibrationRecord]) -> "Calibrator":
        """Pool-adjacent-violators on (confidence, correct), then linear interpolation."""
        pairs = sorted(((r.confidence, 1.0 if r.correct else 0.0) for r in records),
                       key=lambda item: item[0])
        # pool duplicate confidences: one entry per distinct x, weighted by multiplicity
        xs: list[float] = []
        sums: list[float] = []
        counts: list[float] = []
        for conf, y in pairs:
            if xs and abs(conf - xs[-1]) < 1e-12:
                sums[-1] += y
                counts[-1] += 1.0
            else:
                xs.append(conf)
                sums.append(y)
                counts.append(1.0)

        # PAV: merge adjacent blocks while the fit is not non-decreasing
        block_val: list[float] = []   # fitted value of the block
        block_w: list[float] = []     # number of distinct x values in the block
        block_mass: list[float] = []  # sample count in the block (for weighted merges)
        for i in range(len(xs)):
            block_val.append(sums[i] / counts[i])
            block_w.append(1.0)
            block_mass.append(counts[i])
            while len(block_val) > 1 and block_val[-2] > block_val[-1]:
                v1, w1, m1 = block_val.pop(), block_w.pop(), block_mass.pop()
                v0, w0, m0 = block_val.pop(), block_w.pop(), block_mass.pop()
                total_mass = m0 + m1
                block_val.append((v0 * m0 + v1 * m1) / total_mass)
                block_w.append(w0 + w1)
                block_mass.append(total_mass)

        fitted: list[float] = []
        for value, width in zip(block_val, block_w):
            fitted.extend([value] * int(round(width)))
        fitted = fitted[:len(xs)]

        # prune to a minimal monotone breakpoint list
        px, py = [xs[0]], [fitted[0]]
        for x, y in zip(xs[1:], fitted[1:]):
            if abs(y - py[-1]) > 1e-9:
                px.append(x)
                py.append(y)
        if px[-1] != xs[-1]:
            px.append(xs[-1])
            py.append(fitted[-1])
        confs = [_isotonic_lookup(c, px, py) for c in (r.confidence for r in records)]
        return cls(kind="isotonic",
                   isotonic_x=[round(x, 8) for x in px],
                   isotonic_y=[round(y, 8) for y in py],
                   meta={"n": len(records), "fit": "pav", "points": len(px),
                         **evaluate(confs, [r.correct for r in records])})

    # -- metrics on new data -------------------------------------------------
    def evaluate(self, records: Sequence[CalibrationRecord], bins: int = DEFAULT_BINS) -> dict[str, float]:
        confs = [confidence(self.transform(r.distribution)) for r in records]
        return evaluate(confs, [r.correct for r in records], bins=bins)

    def reliability(self, records: Sequence[CalibrationRecord],
                    bins: int = DEFAULT_BINS, adaptive: bool = False) -> list[dict[str, float]]:
        confs = [confidence(self.transform(r.distribution)) for r in records]
        return reliability_table(confs, [r.correct for r in records], bins=bins, adaptive=adaptive)


def _temperature_transform(dist: Mapping[str, float], temperature: float) -> dict[str, float]:
    """softmax(z / T) given p = softmax(z): renormalise `p ** (1/T)`."""
    probs = normalize(dist)
    if temperature == 1.0 or len(probs) < 2:
        return probs
    return normalize({k: v ** (1.0 / temperature) for k, v in probs.items()})


def _isotonic_lookup(conf: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        return clip(conf)
    if len(xs) == 1 or conf <= xs[0]:
        return clip(ys[0])
    if conf >= xs[-1]:
        return clip(ys[-1])
    for i in range(1, len(xs)):
        if conf <= xs[i]:
            span = xs[i] - xs[i - 1]
            t = 0.0 if span <= 0 else (conf - xs[i - 1]) / span
            return clip(ys[i - 1] + t * (ys[i] - ys[i - 1]))
    return clip(ys[-1])


# ------------------------------------------------------- cross-validated fitting
@dataclass
class CalibrationFit:
    """Result of comparing methods: the winner plus the table it was chosen from."""

    calibrator: Calibrator
    table: list[dict[str, Any]]
    selected_by: str
    folds: int
    folds_used: int = 0

    def best_rows(self) -> list[dict[str, Any]]:
        return sorted(self.table, key=lambda row: row["mean"])

    def summary(self, width: int = 22) -> str:
        lines = [f"{'method':<{width}} " + " ".join(f"{m:>12}" for m in ("mean", "std", "raw"))]
        for row in self.best_rows():
            lines.append(f"{row['method']:<{width}} {row['mean']:>12.4f} "
                         f"{row['std']:>12.4f} {row['raw']:>12.4f}")
        return "\n".join(lines)


def fit_calibration(records: Sequence[CalibrationRecord], method: str = "auto",
                    folds: int = 5, seed: int = 0, select_by: str = "ece_adaptive",
                    min_folds: int = 2) -> CalibrationFit:
    """Compare methods by k-fold cross-validation and refit the winner on all data.

    `method="auto"` evaluates identity (raw), temperature, platt and isotonic, and
    picks the lowest mean cross-validated `select_by` score — ECE over equal-count
    bins by default, because equal-width ECE is dominated by the single top bucket
    at these confidence distributions. Ranking metrics (auroc) are printed too:
    all three methods are monotone in the top confidence, so a drop there means
    something is wrong with the fit, not with the method.
    """
    if select_by not in METRICS:
        raise CalibrationError(f"select_by must be one of {METRICS}")
    if not records:
        raise CalibrationError("cannot fit on zero records")
    candidates = METHODS if method == "auto" else (method,)
    n = len(records)
    n_folds = max(min_folds, min(int(folds), n))

    # deterministic, label-stratified fold split
    order = sorted(range(n), key=lambda i: (records[i].correct, _stable_hash(i, seed)))
    folds_idx = [order[i::n_folds] for i in range(n_folds)]

    table: list[dict[str, Any]] = []
    for name in ("identity",) + tuple(candidates):
        scores: list[float] = []
        for fold in folds_idx:
            test = [records[i] for i in fold]
            held_out = set(fold)
            train = [records[i] for i in range(n) if i not in held_out]
            if not test or not train:
                continue
            model = Calibrator(kind="identity") if name == "identity" \
                else Calibrator.fit(train, method=name)
            scores.append(model.evaluate(test)[select_by])
        if not scores:
            continue
        table.append({
            "method": name,
            "mean": mean(scores),
            "std": (sum((s - mean(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
            "raw": Calibrator(kind="identity").evaluate(records)[select_by],
            "folds": len(scores),
        })

    if not table:
        raise CalibrationError("no method could be evaluated")
    raw_row = next((row for row in table if row["method"] == "identity"), None)
    best = min(table, key=lambda row: row["mean"])
    # In auto mode, doing nothing is an option: only ship a calibrator that beats
    # raw softmax out of sample. An explicitly requested method is always fitted.
    if method == "auto" and raw_row is not None and best["method"] != "identity" \
            and best["mean"] >= raw_row["mean"]:
        chosen = "identity"
    else:
        chosen = str(best["method"])

    calibrator = Calibrator(kind="identity") if chosen == "identity" \
        else Calibrator.fit(records, method=chosen)
    calibrator.meta.update({
        "fitted_on": len(records),
        "selected_by": select_by,
        "cv_folds": table[0]["folds"],
        "cv_mean": best["mean"] if chosen != "identity" else (raw_row or {}).get("mean"),
        "cv_table": [{k: v for k, v in row.items()} for row in table],
    })
    return CalibrationFit(calibrator=calibrator, table=table, selected_by=select_by,
                          folds=folds, folds_used=table[0]["folds"])


def _stable_hash(i: int, seed: int) -> int:
    """Small deterministic integer hash (str hashing is randomised per process)."""
    x = (i + 1) * 2654435761 + seed * 40503
    x ^= x >> 13
    x = (x * 1274126177) & 0xFFFFFFFF
    return x
