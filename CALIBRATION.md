# Calibration — making the probabilities in this package usable

Short version: **the engine reports `softmax` over a field's allowed answers, not the
probability that the answer is correct.** This file explains the difference, shows
what we measured, and documents the calibration machinery that now ships with the
package (and what it does and does not fix).

## 1. What the engine computes

For a field like `risk_tier` with choices `LOW / MEDIUM / HIGH / CRITICAL`, the model
produces a logit for every token in its vocabulary at the decision position. We keep
only the tokens that can start an allowed answer and normalise:

```
P(choice i) = exp(z_i) / Σ_j exp(z_j)          # z = logits, temperature 1
```

That is a *conditional next-token probability*: given this context, and given that the
answer starts here, how likely is each allowed continuation. It is a real number from
the model's own distribution — but it answers "which continuation is most likely",
not "how often would this decision be correct".

## 2. What calibration means

A model is calibrated when its stated confidence matches its observed accuracy:

```
P(answer is correct | stated confidence = 0.9) ≈ 0.9
```

Calibration is measured by bucketing answers by confidence and comparing each bucket's
mean confidence with its empirical accuracy. Plotted, a calibrated model lies on the
diagonal (a *reliability diagram*). The summary number is **ECE** — expected
calibration error — the bucket-size-weighted average gap between confidence and
accuracy.

Two variants matter in practice and both ship in `calibration.py`:

- `ece()` — equal-**width** bins. Standard, but here the confidence distribution is
  so skewed (most answers land in one bin) that the number is decided by that single
  bin.
- `adaptive_ece()` — equal-**count** bins, with edges at empirical quantiles so that
  records sharing a confidence never get split across bins. This is the better
  model-selection signal at these distributions, and it is the default (`--select-by
  ece_adaptive`).

A well-calibrated 0.7 means: when this model says 0.7, it is right about 70% of the
time. That property is what lets software act autonomously: act at ≥ threshold, route
the middle band to review, refuse below it.

## 3. Why softmax is not that

Three separate reasons:

1. **Training objective mismatch.** The model was trained to predict the next token in
   a web-scale corpus, not to be right at a rate equal to its probability.
2. **Slicing changes the scale.** We discard everything except a handful of candidate
   tokens and renormalise. Even a well-calibrated full-vocabulary distribution need not
   stay calibrated after conditioning on a small slice.
3. **Chat-tuned models are overconfident.** Preference training rewards confident,
   decisive answers, so stated confidence sits above true accuracy.

## 4. What we measured on the default model

Recorded run, 7B, single pass over the 373 public question slots (`evals/analysis/REPORT.md`):

| slice | n | accuracy | mean confidence (correct) | mean confidence (wrong) |
|---|---|---|---|---|
| `noul` (yes/no) | 236 | 83.1% | 0.959 | 0.891 |
| choice | 110 | 63.6% | — | — |
| score (0–3) | 27 | 25.9% exact, 85.2% within 1 | — | — |

- **ECE ≈ 0.094** (Qwen2.5-7B) equal-width on the earlier noul-only measurement; the
  fuller slice measures **ECE 0.124** on the same basis (n=236 noul).
- The model is **almost always very confident** — 184 of 236 noul answers landed in
  the 0.95–1.00 band — and in that band it was right 88.0% of the time.
- **Wrong answers were nearly as confident as right ones:** 0.891 vs 0.959 mean; 28 of
  40 wrong answers carried ≥0.90.
- **AUROC of the raw confidence is 0.74 overall** (0.91 on choice fields, 0.66 on
  noul). So the score does contain signal about correctness — it is a usable *ranking*
  signal — it is just on the wrong scale.

## 5. What the package now does about it

`parallel_decisions.calibration` implements three post-hoc methods, all fitted on
labelled data and all **monotone in the top confidence**, so they move the reported
number and never the chosen answer:

| method | form | parameters |
|---|---|---|
| `temperature` | `p' ∝ p ** (1/T)` (equivalent to `softmax(z/T)`) | 1 |
| `platt` | `σ(a · logit(c) + b)`, fitted by IRLS | 2 |
| `isotonic` | monotone step fit of confidence → empirical accuracy (PAV) | many |

`fit_calibration()` compares raw confidence against all three by label-stratified
k-fold cross-validation and **refuses to ship a method that loses to raw softmax out
of sample** — "do nothing" is an explicit option.

```bash
.venv/bin/pd calibrate --data labelled.jsonl --out calibration.json
.venv/bin/pd calibrate --data labelled.jsonl --where type=noul   # fit a slice
.venv/bin/python examples/routing.py --data labelled.jsonl --max-error 0.01
```

`Decider(calibration="calibration.json")` then attaches it; every `FieldValue` keeps
`raw_probability` alongside the calibrated `probability`.

### What it fixes, and what it does not

- It fixes the **scale**, which is what thresholding needs. On a 300-row synthetic set
  that is overconfident by construction, temperature scaling moved adaptive ECE from
  0.158 to 0.059 and the reliability table onto the diagonal.
- It does **not** add information. AUROC is unchanged by every method (monotone maps
  cannot reorder); a confidence that does not separate right from wrong cannot be
  calibrated into one that does.
- It does **not** transfer across domains, and often not across question types: a
  calibrator fitted on all field types mixed together can look fine overall while
  being wrong inside a slice, because each slice has its own base rate (`pd calibrate`
  prints the per-slice table for exactly this reason).
- It needs data. On ~100 labelled noul rows from the published eval, **no method beat
  raw softmax out of sample** and `fit_calibration` selected `identity`. That is the
  honest outcome at that sample size, not a failure of the methods: 100 records cannot
  pin a monotone curve at the 0.9+ end where nearly all the mass sits.

We also checked whether a different confidence signal ranks better than the top
probability (`evals/confidence_features.py`): margin over the runner-up, log-odds, and
normalised entropy all land within 0.01 AUROC of the top probability or below it
(0.739 / 0.735 / 0.715 vs 0.743 overall). The top probability is the best of these
features, so the calibrator fits on it.

## 6. Using it as a routing signal

`examples/routing.py` derives the policy from data rather than guessing, and evaluates
it **out of sample** (the calibrator is refit inside the fold loop, so no record is
scored by a calibrator that saw it):

```
error budget -> threshold, coverage, act-bucket error
  budget  threshold  acted on   share  errors  error rate  95% upper
   0.00%     0.8464        11   11.3%       0       0.00%     25.88%
   1.00%     0.8464        11   11.3%       0       0.00%     25.88%
   5.00%     0.8464        11   11.3%       0       0.00%     25.88%

POLICY (budget 1.00%):
  act     conf >= 0.8464   11/97 = 11.3% of work, error rate 0.00% (95% upper 25.88%)
  review  0.5013 <= conf < 0.8464   76 decisions
  refuse  conf < 0.5013   10 decisions
```

Read the **95% upper bound**, not the point estimate: zero errors in 11 decisions is
consistent with a true error rate anywhere up to 26%. A "0.00% error rate" on 11
samples is not evidence of a 0% error rate. The act bucket will only be trustworthy
with a few hundred labelled rows from the domain it runs in — that is the single
thing standing between this and a usable automation threshold.

## 7. Collecting the data

The preferred source is your own domain, because calibration does not transfer:

1. **Your domain samples** — 100–200 labelled items per domain is the stated minimum,
   and the routing table above shows why more is better.
2. **The published eval** (`evals/results/calibration/*.jsonl`), which is what the
   numbers in this file come from. Records carry `distribution`, `chosen`, `target`,
   `correct`, and `type`/`workflow`/`qid` for slicing.
3. **Synthetic cases** (`quality-eval/`) — weakest, but better than nothing for a
   first look.

`evals/domains/<name>/` is the place for the first kind; `evals/domains/invoice/` is
the worked example, and its `score.py` prints each field's accuracy next to the
constant-answer baseline so a high number on a skewed field cannot be mistaken for
skill.
