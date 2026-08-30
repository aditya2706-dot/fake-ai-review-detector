# The project, in phases

What is built, what it costs, and what to do next. Phases 0–5 are **done and
runnable in this repo**; 6+ are the extension path.

---

## Phase 0 — Frame the problem *(done)*

Decide what you are detecting and what "good" means before writing model code.

- Positive class = inauthentic reviews (campaign / incentivised / LLM-written).
- Out of scope: genuine-but-unhelpful reviews.
- Success metric = PR-AUC, plus recall at a 90% precision floor.
- Cost asymmetry: a false accusation is worse than a miss → threshold is a
  policy choice, tiered actions rather than binary delete.

**Deliverable:** `docs/METHODOLOGY.md` §1–2.
**Most common mistake:** optimising accuracy or F1 on an imbalanced problem.

---

## Phase 1 — Data *(done)*

One canonical schema (`src/frd/schema.py`) that every source normalises into,
so nothing downstream cares where the data came from.

| Source | Status |
|---|---|
| Synthetic marketplace generator | `data/synth.py` — 40k reviews, campaigns, label noise |
| LLM-generated fake reviews corpus | `loaders.load_llm_fake_reviews` |
| Amazon review JSONL (unlabelled) | `loaders.load_amazon_jsonl` |
| Any CSV/JSONL/parquet | `loaders.load_generic` with a column map |

`describe_coverage()` reports whether a corpus can support behavioural features
at all — several popular public sets have no reviewer id or timestamp, which
silently reduces the whole system to a text classifier.

**Most common mistake:** training on a corpus with no reviewer/timestamp columns
and reporting the result as if behavioural detection were included.

---

## Phase 2 — Features *(done)*

**31 behavioural** (`features/behavior.py`) — reviewer burstiness, rating
extremity, account tenure, template self-similarity, product review spikes,
rating deviation, fresh-account share. Every one computed strictly
backwards-looking, proven by `tests/test_causality.py`.

**46 linguistic** (`features/text.py`) — the core hypothesis is *specificity*:
real customers cite numbers, measurements, and comparisons ("measured 31cm not
the 29cm listed"); paid ones produce superlative density with no detail.

**Text vectorisers** — TF-IDF word (1,2) plus character (3,5). Character
n-grams survive the paraphrasing farms use to beat duplicate filters.

**Most common mistake:** computing reviewer aggregates over the whole dataset.
This leaks the future into the past and inflates every metric.

---

## Phase 3 — Model *(done)*

Two-stage stack: linear model on sparse text → single score → gradient-boosted
trees over dense behavioural + linguistic features + that score. Stage A's
training-set contribution is out-of-fold, grouped by product.

Ablations are first-class: `text_only`, `linguistic_only`, `behaviour_only`,
`text_plus_linguistic`, `full_fusion` — the only way to know whether the
expensive behavioural machinery earns its cost.

**Most common mistake:** shipping one fused model with no ablation, so nobody
knows which half is doing the work.

---

## Phase 4 — Evaluation *(done)*

- Two split regimes: temporal (production-realistic) and cold-reviewer
  (generalisation to unseen accounts).
- Threshold tuned on validation only, at a target precision.
- Platt calibration so scores are usable as probabilities.
- Slice metrics by category, rating, verified status, text length, and
  reviewer-history depth — the last exposes the cold-start blind spot the
  headline number hides.
- Negative controls: shuffled labels must produce no signal.

**Most common mistake:** tuning the threshold on test.

---

## Phase 5 — Serving *(done)*

FastAPI service with `/score`, `/score/batch`, `/health`, `/model`. Explanations
in three layers: exact tree-SHAP feature contributions, exact linear token
contributions, and curated English reason codes for moderators.

The service **reuses the training feature code** rather than reimplementing it,
and `tests/test_api.py::test_api_and_offline_scores_agree` asserts both paths
produce the same number.

**Most common mistake:** a second feature implementation on the serving path.
Offline metrics stay perfect while the live model quietly degrades.

---

## Phase 6 — Real labelled data *(next, highest value)*

Everything above is validated on simulated data. It proves the plumbing, not the
performance. Options in descending order of value:

1. **Your own moderation logs.** Confirmed-fraud decisions are the best labels
   available. Expect noisy, biased-toward-obvious positives.
2. **Enforcement actions as weak labels.** Reviews from banned accounts are
   presumptive positives — cheap, plentiful, and biased toward farms you already
   catch.
3. **Public corpora** for the text half only, with the coverage caveat above.
4. **Human annotation** of a stratified sample, for a trustworthy test set even
   when training on weak labels.

Then re-run the ablation. If `behaviour_only` does not beat `text_only` on real
data as it does here, the behavioural features are not transferring and that is
the finding.

---

## Phase 7 — Graph features

The current model reasons about one account and one product at a time. Farms
that spread activity thinly beat per-account thresholds. Bipartite
reviewer↔product structure catches them:

- Reviewer co-review graph; dense subgraph / community detection.
- Node2vec or GNN embeddings over the bipartite graph.
- Temporal co-occurrence: accounts that repeatedly appear together within
  short windows.

Expect the largest single lift after real labels, especially on the cold-start
slice, because a new account inside a known cluster is suspicious on arrival.

---

## Phase 8 — Transformer text stage

`models/transformer.py` is written and lazily imported (`pip install -e ".[deep]"`).
Worth it in two specific cases: **LLM-generated reviews** (fluent, template-free,
so n-gram overlap is low) and **multilingual marketplaces**. Not worth it if the
ablation shows behaviour dominating — measure before paying for GPU latency.

---

## Phase 9 — Production concerns

| Concern | Approach |
|---|---|
| Retraining | Scheduled (monthly) + triggered on drift; tactics change |
| Drift monitoring | PSI on feature distributions, flag-rate alarms, weekly precision audit on a sampled queue |
| Feature store | Behavioural aggregates precomputed and cached; recomputing history per request will not hold latency |
| Shadow mode | Score without acting for weeks; compare to moderator decisions before enforcement |
| Appeals | Every automated removal must be appealable, with the reason codes attached |
| Adversarial review | Assume farms read your published signals; keep some features unpublished and expensive to evade |
| A/B measurement | Ultimate metric is not PR-AUC but marketplace trust: return rates, review-helpfulness votes, complaint volume |

---

## Phase 10 — Fairness and governance

Check flag rates by seller size, product category, reviewer geography, and
non-native-English writing style. **Terse, unhedged, superlative-heavy text is
also what a non-native speaker writes** — the specificity features risk a
systematic bias against them. Measure it explicitly; if present, reweight or
drop the offending features. This is a real risk in this exact design, not a
box-ticking exercise.
