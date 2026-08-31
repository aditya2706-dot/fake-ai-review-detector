# Model card — Fake Review Detector v0.1.0

## Overview

| | |
|---|---|
| **Task** | Binary classification: is this e-commerce review inauthentic? |
| **Output** | Calibrated P(fake) in [0,1], plus a tiered action and reason codes |
| **Architecture** | Two-stage stack — TF-IDF + LogisticRegression → `text_score`; gradient-boosted trees over behavioural + graph + linguistic + `text_score` |
| **Training data** | Synthetic marketplace, 40k reviews, ~19% positive. **Not production data.** |
| **Test PR-AUC** | 0.914 temporal / 0.905 cold-reviewer, against a 0.172 base rate |
| **Primary metric** | PR-AUC on a temporal split |
| **Version** | 0.1.0 |

## Intended use

**In scope.** Ranking reviews for human moderation on an e-commerce
marketplace; prioritising a finite moderation queue; flagging products with
anomalous review patterns for investigation; as one input among several in a
trust-and-safety decision.

**Out of scope.**

- **Fully automated removal without appeal.** The `auto_remove` tier exists for
  high-confidence cases, but every automated action must be appealable with the
  reason codes attached.
- **Judging individual people.** A high score describes a *review* in a
  *context*, not a person's honesty. Do not use it to ban accounts on its own.
- **Any legal, employment, or credit decision.**
- **Detecting genuine-but-negative reviews.** Suppressing criticism is the
  failure mode this model must never be used for. It is trained on
  inauthenticity, not sentiment, and the linguistic features treat hedging and
  concrete complaints as evidence *for* authenticity.

## Performance

Reported by `make train` into `artifacts/reports/training_report.json`, with
per-slice breakdowns. Two split regimes are always reported: **temporal**
(production-realistic) and **cold-reviewer** (unseen accounts). PR-AUC is
reported against the base rate so the real lift is visible.

**These figures are on synthetic data.** They validate correctness of the
pipeline — causal features, no split leakage, no train/serve skew — not
real-world performance. Re-measure on labelled production data before trusting
any number for planning.

**The generator was revised during development** — farms given reusable account
rosters, farm accounts given genuine cover traffic — which moved every reported
figure. `docs/METHODOLOGY.md` §5.1 publishes the before and after side by side.

**Known blind spot, stated plainly:** the model recalls **nothing** at the
operating threshold for authors with fewer than three prior reviews. Graph
features were built specifically to address this and did not
(`docs/ROADMAP.md` phase 7). Any deployment must route first-time reviewers to
a policy that does not depend on this model.

## Factors examined

`evaluate.slice_metrics` reports PR-AUC, precision, and recall for:

- product category
- star rating (1 / 2–3 / 4 / 5)
- verified-purchase status
- review text length
- **reviewer history depth (0 / 1–2 / 3–9 / 10+ prior reviews)** — the
  cold-start slice, where behavioural features are sentinels and performance is
  weakest

## Limitations

1. **Cold start.** No author or product history → text-only performance.
2. **Concept drift.** Fraud tactics evolve; a stale model decays. Retrain on a
   schedule and monitor flag-rate and feature drift.
3. **Adversarial adaptation.** Documenting the features tells farms what to
   avoid. Behavioural features are the more robust half because evading them
   costs real money — more accounts, slower posting, genuine purchases.
4. **Label ceiling.** Trained on moderator-style labels, which are noisy and
   biased toward easy-to-spot fakes. The model inherits that bias and will be
   weakest on the sophisticated fakes nobody labelled.
5. **Single-language.** English lexicons and TF-IDF vocabulary. Other languages
   need retraining, or the multilingual transformer stage.
6. **No graph structure.** Farms that spread activity thinly across accounts and
   time beat per-account thresholds.

## Ethical considerations

**Language bias is the main risk, and it is specific.** The linguistic features
reward concrete detail and hedging, and penalise superlative density and terse
generic praise. That is a reasonable proxy for paid text — and also a
description of how many **non-native English speakers** write genuine reviews.
Deploying without measuring flag rates across writing-style and geography
proxies risks systematically suppressing their reviews.

**Mitigations.**

- Audit flag rates by geography, seller size, and a non-native-writing proxy
  before enforcement, and on a schedule after.
- Prefer behavioural features, which are not language-dependent, when a
  trade-off must be made.
- Run in shadow mode first: score without acting, compare against moderator
  decisions, and inspect disagreements.
- Keep humans deciding on anything short of overwhelming evidence.
- Preserve appeals with reason codes, and feed appeal outcomes back as labels.

**Asymmetric harm.** A missed fake review slightly degrades marketplace
integrity. A wrongly removed genuine review silences a real customer, and at
scale erodes exactly the trust the system exists to protect. The threshold
policy encodes that asymmetry deliberately; do not raise recall by lowering the
precision floor without an explicit decision from whoever owns that trade-off.

## Maintenance

| Item | Cadence |
|---|---|
| Retraining | Monthly, or on drift alarm |
| Precision audit on a sampled queue | Weekly |
| Fairness audit across slices | Quarterly |
| Feature drift (PSI) monitoring | Continuous |
| Model card review | Every retrain |
