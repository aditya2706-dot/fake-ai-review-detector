# The project, in phases

What is built, what it costs, and what to do next. Phases 0–5 and 7 are **done
and runnable in this repo**; the rest are the extension path. Phase 7's entry
below is kept as a record of a prediction this document got wrong.

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
| Synthetic marketplace generator | `data/synth.py` — 40k reviews, farm rosters, cover traffic, campaigns, label noise |
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

**18 graph** (`features/graph.py`) — co-review neighbourhood overlap normalised
for activity and product popularity, strong-tie components, cohort cliquishness,
temporal co-arrival. Streaming, and proven causal by `tests/test_graph.py`. See
phase 7 for what it did and did not deliver.

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
`graph_only`, `text_plus_linguistic`, `full_fusion`, `behaviour_plus_graph`,
`full_fusion_plus_graph` — the only way to know whether the expensive
behavioural and graph machinery earns its cost. `full_fusion` is deliberately
kept as the pre-graph combination so the graph block's contribution is a
subtraction between two rows of one table.

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

Two surfaces sit on top of it (`web/`), both reading one scored-corpus cache: a
moderation console, and a storefront that shows every rating as listed and again
with the `auto_remove` tier withheld. The storefront is where the cost asymmetry
stops being theoretical -- it withholds at the auto-remove line rather than the
queue threshold, which is the difference between suppressing an innocent review
one time in eight and one time in seventeen. See README "The storefront".

**Most common mistake:** a second feature implementation on the serving path.
Offline metrics stay perfect while the live model quietly degrades.

---

## Phase 6 — Real labelled data *(next, and now the only thing that can settle phase 7)*

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

## Phase 7 — Graph features *(done; the prediction below was wrong)*

**What this section originally predicted:** *"Expect the largest single lift
after real labels, especially on the cold-start slice, because a new account
inside a known cluster is suspicious on arrival."*

**What measuring it produced, first time:** +0.004 test PR-AUC on the fused
model, 5.1% of stage-B feature importance, and **no improvement at cold
start**.

**What measuring it four times produced:** nothing distinguishable from zero.
Across two corpus draws × two split regimes the block's contribution to the
fused model reads **−0.0002, +0.0038, +0.0018, −0.0033** — all inside 0.004,
sign effectively random. Validation and test disagreed about whether to ship
it on *both* corpora, in opposite directions, which is how the
selection-on-test bug in METHODOLOGY §3.6 got found: the presence of this
entire feature block in the shipped model was a test-set coin flip. It stays
in because validation prefers it on the current corpus, which is a much weaker
claim than "+0.004 PR-AUC" sounded.

Cold start is unchanged and unhelped: the `reviewer_history=0` slice reads
0.017–0.038 across draws on 217–225 rows with single-digit positives, which is
noise around nothing. The prediction was wrong in a way that is obvious in hindsight: *an
account with no history has no edges either.* The co-review graph is exactly as
empty at cold start as the behavioural counters are, and for the same reason.
"A new account inside a known cluster" presumes you can place the account in
the cluster, which is precisely what you cannot do on its first review.

**What was built** (`src/frd/features/graph.py`, 18 features):

- neighbourhood overlap per partner, normalised two ways — by how much each
  account posts (Jaccard) and by how obscure the shared products were
  (Adamic–Adar). The normalisation is not a refinement, it is the feature:
  counted raw, overlap measures enthusiasm and comes out *anti*-correlated with
  fraud at AUC 0.36;
- strong-tie components via streaming union-find, admitted at two shared
  products and a surprise threshold, because on the raw co-review graph the
  whole marketplace collapses into one component within a few thousand rows;
- cohort cliquishness for the product being reviewed — are its recent reviewers
  strangers, or have they all met before;
- temporal co-arrival: accounts that keep landing inside the same 48-hour
  window.

**Streaming statistics, not embeddings.** This section used to say "node2vec or
GNN embeddings". That is temporal leakage — see `docs/METHODOLOGY.md` §3.3. An
embedding fitted to the finished graph hands every January review a summary of
its own future. Every feature above is instead maintained incrementally in one
chronological pass and passes the same truncation proof as the behavioural
block. Embeddings stay reachable through periodic snapshots, each fitted only
to data before its cutoff — a feature-store project, not a function.

**It also exposed a defect in the simulator** (the first of two; the second,
a sentiment/rating oracle worth 97.9% precision to a hand-written rule, is
METHODOLOGY §5.2). The graph block initially found
nothing because the generator contained no coordinated structure: campaigns
drew workers independently from a flat account pool, so farm pairs co-reviewed
*less* than genuine pairs did (0.8% vs 6.8%), and farm accounts never posted
cover reviews, making the account id 98% predictive of the label. Both are now
fixed and both sets of numbers are published in `docs/METHODOLOGY.md` §5.1.
Finding this was worth more than the feature block was.

**Where the graph should still pay**, and what would settle it:

1. **Farms that pace themselves below burst thresholds.** On this corpus a
   campaign *is* a burst, so the behavioural features already catch what the
   graph would. A farm posting one review per account per week defeats them and
   leaves the co-review structure intact. The synthetic generator does not
   currently simulate one — that, not more graph features, is the next honest
   experiment, and it is now the *only* remaining way to justify keeping the
   block: four measurements have failed to show it helping.
2. **The cold-reviewer split**, where `graph_only` scores 0.567 against 0.550
   on the temporal split. Structure still generalises to unseen accounts a
   little better than it generalises across time — the one result that came out
   as predicted — but the gap was 0.079 on the first corpus draw and is 0.017
   on the second, so treat the direction as the finding and not the size.
3. **Real labelled data** (phase 6), where farms are vastly more structured
   than any simulation of them.

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
