# Calibration vs softmax — what the probabilities in this package actually mean

Short version: **the numbers here are softmax scores, not calibrated probabilities.**
They are useful for ranking answers inside a field. They are not safe to threshold on
("only act if probability > 0.9") without checking them first. This file explains the
difference, shows what we measured, and lists the standard fixes.

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
diagonal (a *reliability diagram*). The summary number is **ECE** — expected calibration
error — the bucket-size-weighted average gap between confidence and accuracy.

A well-calibrated 0.7 means: when this model says 0.7, it is right about 70% of the
time. That property is what lets software act autonomously: threshold at 0.95 for
automation, 0.6–0.95 for human review, below 0.6 for "ask again / escalate".

## 3. Why softmax is not that

Three separate reasons:

1. **Training objective mismatch.** The model was trained to predict the next token in
   a web-scale corpus, not to be right at a rate equal to its probability. Next-token
   likelihood and answer correctness are correlated but distinct quantities.
2. **Slicing changes the scale.** We discard everything except a handful of candidate
   tokens and renormalise. Even a perfectly calibrated full-vocabulary distribution
   need not stay calibrated after conditioning on a small slice.
3. **Chat-tuned models are usually overconfident.** Preference training rewards
   confident, decisive answers and penalises hedging, so stated confidence tends to
   sit above true accuracy. This is well documented across open and closed models.

## 4. What we measured on this package's default model

From the full public TypeSafe evaluation (Noul questions, where the run recorded the
per-answer probability; 236 scored pairs, one run):

| confidence bucket | n | accuracy | gap |
|---|---|---|---|
| 0.5–0.6 | 6 | 66.7% | +11.7 |
| 0.6–0.7 | 8 | 62.5% | −2.5 |
| 0.7–0.8 | 9 | 44.4% | −30.6 |
| 0.8–0.9 | 9 | 77.8% | −7.2 |
| 0.9–1.0 | 204 | 86.3% | −8.7 |

- **ECE ≈ 0.094** (Qwen2.5-7B); the 8B variant measured 0.142.
- The model is **almost always very confident** — 204 of 236 answers landed in the
  0.9–1.0 bucket — and in that bucket it was right 86.3% of the time.
- **Wrong answers were nearly as confident as right ones:** mean confidence 0.89 on
  incorrect vs 0.96 on correct; 28 of 40 wrong answers carried ≥0.90 confidence.

Practical consequence: on this workload the score works as a *ranking* signal within a
question (which of these options is most likely), but **not** as a *decision* signal
("should I act on this answer?"). Thresholding on it would let a large share of errors
straight through.

Caveats: single run, curated evaluation cases, n=236, and only Noul questions (the
only ones where the runner recorded an answer probability). The heavy skew into the
top bucket means ECE is dominated by that one row.

## 5. What "trained calibration" (TypeSafe's RLCD) changes

TypeSafe's claim is that they optimise for calibration directly — the training signal
rewards saying 0.7 when the model is right ~70% of the time, not just being right.
If that works, the reliability diagram hugs the diagonal and confidence becomes a
usable routing signal (their docs build directly on this: autonomy thresholds,
escalation bands, "confidence-gated routing").

This package cannot recreate that without training. Its default model was never
optimised for calibration, so the honest position is: treat probabilities as relative
confidence, and calibrate them yourself before using them in control flow.

## 6. Fixes that do not require retraining

1. **Temperature scaling** — fit one parameter `T` on labelled data so that
   `P' = softmax(z / T)` minimises ECE. This is the standard first step (Guo et al.,
   "On Calibration of Modern Neural Networks"); it preserves argmax, so decisions do
   not change, only the reported confidence.
2. **Platt scaling / isotonic regression** — map raw confidence to empirical accuracy
   with a monotone fit. More flexible than temperature; needs more labelled data.
3. **Bucket and threshold** — using the measured reliability table above, act on the
   0.9+ bucket only for high-stakes automation, and route the rest to review. Crude
   but effective, and it uses the data you already have.
4. **Collect your own labelled set per domain** — calibration is domain-specific; a
   curve fitted on invoice data does not transfer to support tickets.

If you want temperature scaling built into this package, the honest interface would be
`Decider(calibration="path/to/temperature.json")`, with a small script to fit `T` from
a labelled set. Nothing is implemented yet — the current code reports raw softmax and
says so in `README.md`.
