# Methodology

This document records the decisions that determine whether the reported numbers
mean anything. Most of them are about avoiding ways to accidentally cheat.

---

## 1. What counts as a "fake" review

The positive class is **inauthentic reviews**, covering three quite different
behaviours:

| Type | Description | Dominant signal |
|---|---|---|
| Coordinated campaign | A seller buys a burst of reviews for one product from a farm | Behavioural: burstiness, account reuse, template overlap |
| Lone incentivised | An individual paid or refunded for a positive review | Text + weak account signals |
| Machine-generated | LLM-written reviews, at scale | Distributional text properties |

Lumping them into one label is a modelling compromise worth being explicit
about: each has a different detectable signature, and a single-threshold model
will be better at some than others. The per-slice metrics exist to show which.

Deliberately **out of scope**: reviews that are genuine but unhelpful, angry, or
mistaken. Filtering those is a content-quality problem, not a fraud problem, and
conflating them is how a detector starts deleting legitimate criticism.

---

## 2. Why the headline metric is PR-AUC, not accuracy

At a ~19% fake rate, a model that flags nothing scores 81% accuracy. Accuracy
is not just uninformative here, it is actively misleading.

- **PR-AUC (average precision)** — ranking quality for the rare class. Reported
  against the base rate, so `lift_over_random` shows the real gain.
- **ROC-AUC** — reported for comparability with the literature, but it flatters
  imbalanced problems: the huge negative class keeps the false-positive rate
  small even when most flags are wrong.
- **Precision@k** — if moderators can review 2% of daily volume, what fraction
  of that 2% is actually fake? This is the number that sets staffing.
- **Recall at a precision floor** — how much fraud is caught while keeping
  false accusations inside what the business tolerates.

### The threshold is a policy decision, not 0.5

Deleting a real customer's review costs trust, generates a support ticket, and
occasionally a public complaint. Missing a fake review costs a little
marketplace integrity. These are not symmetric, and nothing about 0.5 reflects
that ratio. So the threshold is chosen as *the lowest value whose validation
precision still meets a target* (default 0.90) — maximum recall subject to a
false-accusation ceiling. `threshold_table()` prints the whole frontier in
moderator-workload terms.

The system emits **three tiers**, not a binary verdict:

| Tier | Condition | Action |
|---|---|---|
| `auto_remove` | score ≥ threshold + 0.25 | Remove, log, allow appeal |
| `queue_for_review` | score ≥ threshold | Human moderator decides |
| `allow` | below threshold | Publish |

---

## 3. Leakage: the two mistakes that would invalidate everything

### 3.1 Temporal leakage in behavioural features

Behavioural aggregates are the strongest features and the easiest to compute
wrongly. "This reviewer's average rating", computed over the whole dataset,
encodes the reviewer's *future* behaviour into a past review. Offline metrics
improve; production performance does not.

**The rule enforced here:** a feature for review *i* may use only reviews
strictly earlier than *i*. Implementation is one chronological pass with
running per-reviewer and per-product state (`features/behavior.py`).

Because every feature looks strictly backwards, features are computed over the
**whole corpus before splitting**. This is not leakage — it is what production
does, where the scorer queries a live reviews table containing all history up to
now. A test-period review sees train-period history and nothing after its own
timestamp.

**This is proven, not asserted.** `tests/test_causality.py` recomputes features
on a corpus truncated at row *i* and asserts row *i*'s feature vector is
identical to its value from the full corpus. Any feature that peeked forward
makes the two differ and fails the test.

### 3.2 Split leakage

A random split puts reviews from the *same campaign* on both sides. The model
then recognises a template it was trained on, and PR-AUC overstates reality
substantially. No random split is offered by this project. Two regimes are
reported instead:

- **Temporal** (70/15/15 chronological) — the production-realistic number.
  Fraud tactics drift, so this split is appropriately pessimistic. The shipped
  model is selected on it.
- **Cold-reviewer** (disjoint reviewer sets) — does the model generalise to
  accounts it has never seen, or has it memorised known-bad reviewer ids? A
  model that looks strong temporally but collapses here is a blocklist wearing a
  model's clothes.

### 3.3 Negative controls

`tests/test_leakage.py` asserts the pipeline *cannot* find signal that does not
exist: shuffled labels must collapse PR-AUC to the base rate, and identifier
columns must never reach the model. Without tests like these, a leakage bug
reads as a great result.

---

## 4. Architecture: why two stages

The two signal families have incompatible shapes.

| Family | Size | Best learner |
|---|---|---|
| Text (TF-IDF word + char) | ~100k sparse dims | Linear — fast, strong, interpretable |
| Behaviour + linguistic | ~77 dense, interaction-rich | Gradient-boosted trees |

Concatenating them forces one learner to work on data it is bad at. So:

```
Stage A   TF-IDF word(1,2) + char_wb(3,5)  ->  LogisticRegression  ->  text_score
Stage B   GBDT over [ 31 behavioural | 46 linguistic | text_score ]
```

Stage A's contribution to Stage B's training data is **out-of-fold, grouped by
product**. Plain k-fold would leave a campaign's reviews in several folds, so
Stage A would recognise its template from the others and hand Stage B an
unrealistically good score. Grouping by product keeps a campaign inside one
fold.

### Why character n-grams

Review farms paraphrase and inject typos specifically to defeat
exact-duplicate filters. Word n-grams break under those edits; character
n-grams survive them.

### Calibration

Stage B trains with positive-class sample weights — necessary to make it attend
to the rare class, but it inflates raw output, so a raw 0.8 does not mean an 80%
chance of fake. A Platt (sigmoid) calibrator fitted on validation fixes the
probability scale. Ranking metrics are unchanged by a monotone map; what changes
is that the tiers above become meaningful. Platt rather than isotonic because a
validation split has few positives and isotonic overfits them.

---

## 5. The synthetic data problem, and how it is handled honestly

Public corpora are small, awkwardly licensed, or missing the reviewer/timestamp
columns the strongest features need. So the default dataset is simulated.

**The risk is obvious:** a naive generator makes fakes trivially separable
(fake = template, genuine = free text), the model scores 0.99, and nothing has
been measured. `data/synth.py` therefore fights itself deliberately:

- both classes draw from **one shared vocabulary pool**;
- ~30% of fakes are **high-effort**: specific, varied, sometimes verified via a
  refund-for-review scheme — catchable only behaviourally;
- ~10% of genuine reviews are **brand-loyalist gushing** that reads like an ad —
  the false-positive trap;
- **lone-wolf fakes** have no campaign structure, so no burst signal at all;
- a substantial share of fakes carry **deliberately moderate 3–4 star ratings**.
  This is a real tactic: a campaign of pure 5s shows up in any
  rating-histogram monitor, so sophisticated farms mix in measured-sounding
  mid-ratings. An earlier version of this generator omitted it, and the effect
  was visible — the 2–3 star slice had no positives to learn from, and the
  model was implicitly treating "extreme rating" as near-sufficient. Adding the
  tactic lowered headline PR-AUC and made the problem honest;
- **2% label noise**, because real ground truth comes from human moderators who
  are themselves wrong sometimes.

The result lands in a realistic band rather than a suspicious 0.99.

**What synthetic data can and cannot tell you.** It validates that the pipeline
is correct, the features are causal, the architecture fuses signal, and the
serving path matches training. It does **not** tell you the model will hit these
numbers on your marketplace — real adversaries adapt, and real text is messier.
Treat the numbers as an upper bound on a lower bound: the plumbing works, and
re-measurement on labelled production data is step one of deployment.

---

## 6. Known limitations

1. **Single-review scoring is the weak case.** With no author or product
   history, behavioural features degrade to sentinels and only text remains.
   The `reviewer_history=0 (cold)` slice quantifies this. Graph-level detection
   (below) is the real answer.
2. **Concept drift.** Fraud tactics change; a model trained on last quarter's
   campaigns decays. Needs scheduled retraining and drift monitoring.
3. **Adversarial pressure.** Publishing which features matter tells farms what
   to avoid. Behavioural features are the more robust half precisely because
   evading them costs real money (more accounts, slower posting).
4. **Label quality is the ceiling.** Moderator labels are noisy and
   systematically biased toward the fakes that are easy to spot; the model
   inherits that bias.
5. **No graph features yet.** Reviewer-product bipartite structure catches
   farms that spread activity thinly enough to beat per-account thresholds.
   Highest-value next addition.
