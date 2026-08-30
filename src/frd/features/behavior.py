"""Causal behavioural feature engineering.

THE CENTRAL IDEA
----------------
Fake reviews are hard to spot one at a time and easy to spot in aggregate.
A single glowing 5-star post is unremarkable; forty of them on one product
inside six hours, from accounts that each post four reviews a day, is not.
So the strongest features here describe the *context* a review arrived in,
not its words.

THE CENTRAL TRAP
----------------
Those context features are exactly where fake-review projects leak. If you
compute "this reviewer's average rating" over the whole dataset, the value for
a review in January already encodes the reviewer's December behaviour -- the
future. Metrics go up, production performance does not. The published
literature is full of this.

So every feature in this module obeys one rule:

    A feature for review i may only use reviews strictly earlier than i.

Implementation is a single chronological pass with running per-reviewer and
per-product state. Because each row looks only backwards, the features can be
computed once over the entire corpus BEFORE splitting -- a test-set review
legitimately sees train-period history, exactly as it would in production,
where the model queries a live reviews table. What it never sees is anything
after its own timestamp.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from ..schema import (
    CATEGORY, PRODUCT_ID, RATING, REVIEWER_ID, TEXT, TIMESTAMP, VERIFIED,
)

# How many prior texts to retain per reviewer / product for similarity checks.
# Campaign templates repeat inside a short window, so a bounded window catches
# them at a fraction of the cost of comparing against all history.
SIM_WINDOW = 25

# Hashed unigram space for cheap cosine similarity. 2**13 keeps collisions
# negligible at review-length texts while staying tiny in memory.
_HASHER = HashingVectorizer(
    n_features=2 ** 13,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    ngram_range=(1, 2),
    dtype=np.float32,
)

BEHAVIOUR_COLUMNS: list[str] = [
    # --- reviewer history ---
    "rv_prior_count",
    "rv_tenure_days",
    "rv_rate_per_day",
    "rv_prior_mean_rating",
    "rv_prior_std_rating",
    "rv_prior_extreme_share",
    "rv_hours_since_prev",
    "rv_count_24h",
    "rv_count_7d",
    "rv_max_burst_24h",
    "rv_distinct_products",
    "rv_product_repeat_share",
    "rv_verified_share",
    "rv_mean_text_len",
    "rv_text_len_zscore",
    "rv_self_similarity_max",
    "rv_self_similarity_mean",
    "rv_category_entropy",
    "rv_night_share",
    # --- product context ---
    "pr_prior_count",
    "pr_prior_mean_rating",
    "pr_rating_deviation",
    "pr_prior_five_share",
    "pr_count_24h",
    "pr_count_7d",
    "pr_burst_ratio",
    "pr_hours_since_prev",
    "pr_verified_share",
    "pr_peer_similarity_max",
    "pr_peer_similarity_mean",
    "pr_new_reviewer_share_7d",
]


@dataclass(slots=True)
class _ReviewerState:
    """Running per-reviewer state.

    Ratings and text lengths are kept as running sums rather than lists.
    Recomputing mean/std from a growing list on every row is O(n) per row and
    therefore quadratic in a prolific reviewer's history -- fine on a 40k
    sample, a real problem on a marketplace where the top accounts have tens of
    thousands of reviews. Sums give the same numbers in O(1).
    """

    n: int = 0
    rating_sum: float = 0.0
    rating_sq_sum: float = 0.0
    extreme_count: int = 0
    len_sum: float = 0.0
    len_sq_sum: float = 0.0
    times: Deque[float] = field(default_factory=deque)          # 24h window
    times_7d: Deque[float] = field(default_factory=deque)       # 7d window
    texts: Deque[sparse.csr_matrix] = field(default_factory=lambda: deque(maxlen=SIM_WINDOW))
    products: dict[str, int] = field(default_factory=dict)
    verified: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    night_count: int = 0
    first_time: float | None = None
    max_burst_24h: int = 0


@dataclass(slots=True)
class _ProductState:
    ratings: list[float] = field(default_factory=list)
    times: Deque[float] = field(default_factory=deque)
    texts: Deque[sparse.csr_matrix] = field(default_factory=lambda: deque(maxlen=SIM_WINDOW))
    verified: int = 0
    five_count: int = 0
    new_reviewer_events: Deque[tuple[float, int]] = field(default_factory=deque)


def _trim(dq: Deque[float], now: float, window_s: float) -> None:
    while dq and now - dq[0] > window_s:
        dq.popleft()


def _entropy(counts: dict[str, int]) -> float:
    if not counts:
        return 0.0
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    p = np.fromiter((c / total for c in counts.values()), dtype=np.float64)
    return float(-(p * np.log(p + 1e-12)).sum())


def _sim_stats(
    vec: sparse.csr_matrix, history: Deque[sparse.csr_matrix]
) -> tuple[float, float]:
    """Max and mean cosine similarity of `vec` against recent history.

    Vectors are already L2-normalised by the hasher, so a dot product IS the
    cosine. Near-1.0 max similarity means a reused template.
    """
    if not history:
        return 0.0, 0.0
    stacked = sparse.vstack(history, format="csr")
    sims = (stacked @ vec.T).toarray().ravel()
    return float(sims.max()), float(sims.mean())


ONE_DAY = 86_400.0
SEVEN_DAYS = 7 * ONE_DAY


def build_behaviour_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Return a frame of causal behavioural features aligned to `df`'s index.

    `df` must be sorted by timestamp ascending (schema.normalise guarantees it).
    """
    ts = df[TIMESTAMP]
    if not ts.is_monotonic_increasing:
        raise ValueError("input must be sorted by timestamp; call schema.normalise")

    texts = df[TEXT].fillna("").astype(str).tolist()
    text_matrix = _HASHER.transform(texts)          # one vectorisation pass

    unix = (ts.astype("int64") // 10 ** 9).to_numpy().astype(np.float64)
    ratings = df[RATING].to_numpy(dtype=np.float64)
    reviewers = df[REVIEWER_ID].astype(str).to_numpy()
    products = df[PRODUCT_ID].astype(str).to_numpy()
    cats = df[CATEGORY].astype(str).to_numpy()
    verified = df[VERIFIED].fillna(False).astype(bool).to_numpy()
    hours = ts.dt.hour.to_numpy()
    text_lens = np.array([len(t) for t in texts], dtype=np.float64)

    n = len(df)
    out = np.zeros((n, len(BEHAVIOUR_COLUMNS)), dtype=np.float32)
    col = {name: i for i, name in enumerate(BEHAVIOUR_COLUMNS)}

    rv_state: dict[str, _ReviewerState] = defaultdict(_ReviewerState)
    pr_state: dict[str, _ProductState] = defaultdict(_ProductState)

    for i in range(n):
        t = unix[i]
        rv = rv_state[reviewers[i]]
        pr = pr_state[products[i]]
        vec = text_matrix[i]

        # ---------------- reviewer features (from prior state only) --------
        prior_n = rv.n
        out[i, col["rv_prior_count"]] = prior_n

        if rv.first_time is not None:
            tenure_days = (t - rv.first_time) / ONE_DAY
            out[i, col["rv_tenure_days"]] = tenure_days
            out[i, col["rv_rate_per_day"]] = prior_n / max(tenure_days, 1.0 / 24)
        else:
            out[i, col["rv_tenure_days"]] = 0.0
            out[i, col["rv_rate_per_day"]] = 0.0

        if prior_n:
            mean_r = rv.rating_sum / prior_n
            out[i, col["rv_prior_mean_rating"]] = mean_r
            var_r = max(rv.rating_sq_sum / prior_n - mean_r * mean_r, 0.0)
            out[i, col["rv_prior_std_rating"]] = np.sqrt(var_r) if prior_n > 1 else 0.0
            out[i, col["rv_prior_extreme_share"]] = rv.extreme_count / prior_n
            out[i, col["rv_hours_since_prev"]] = (t - rv.times[-1]) / 3600.0 if rv.times else -1.0
            out[i, col["rv_verified_share"]] = rv.verified / prior_n
            mean_len = rv.len_sum / prior_n
            out[i, col["rv_mean_text_len"]] = mean_len
            var_len = max(rv.len_sq_sum / prior_n - mean_len * mean_len, 0.0)
            sd = np.sqrt(var_len)
            out[i, col["rv_text_len_zscore"]] = (
                (text_lens[i] - mean_len) / sd if sd > 1e-6 else 0.0
            )
            out[i, col["rv_distinct_products"]] = len(rv.products)
            out[i, col["rv_product_repeat_share"]] = 1.0 - len(rv.products) / prior_n
            out[i, col["rv_category_entropy"]] = _entropy(rv.categories)
            out[i, col["rv_night_share"]] = rv.night_count / prior_n
        else:
            # First-ever review for this account. -1 sentinels let the tree
            # split "no history" apart from "history with value 0", which a
            # 0-fill would silently merge.
            out[i, col["rv_hours_since_prev"]] = -1.0
            out[i, col["rv_prior_mean_rating"]] = -1.0
            out[i, col["rv_verified_share"]] = -1.0
            out[i, col["rv_mean_text_len"]] = -1.0
            out[i, col["rv_category_entropy"]] = -1.0
            out[i, col["rv_night_share"]] = -1.0

        _trim(rv.times, t, ONE_DAY)
        c24 = len(rv.times)
        out[i, col["rv_count_24h"]] = c24
        rv.max_burst_24h = max(rv.max_burst_24h, c24)
        out[i, col["rv_max_burst_24h"]] = rv.max_burst_24h

        # 7-day count keeps its own deque so trimming one window never
        # disturbs the other.
        _trim(rv.times_7d, t, SEVEN_DAYS)
        out[i, col["rv_count_7d"]] = len(rv.times_7d)

        smax, smean = _sim_stats(vec, rv.texts)
        out[i, col["rv_self_similarity_max"]] = smax
        out[i, col["rv_self_similarity_mean"]] = smean

        # ---------------- product features --------------------------------
        p_prior = len(pr.ratings)
        out[i, col["pr_prior_count"]] = p_prior
        if p_prior:
            parr = np.asarray(pr.ratings)
            pmean = parr.mean()
            out[i, col["pr_prior_mean_rating"]] = pmean
            out[i, col["pr_rating_deviation"]] = abs(ratings[i] - pmean)
            out[i, col["pr_prior_five_share"]] = pr.five_count / p_prior
            out[i, col["pr_verified_share"]] = pr.verified / p_prior
            out[i, col["pr_hours_since_prev"]] = (
                (t - pr.times[-1]) / 3600.0 if pr.times else -1.0
            )
        else:
            out[i, col["pr_prior_mean_rating"]] = -1.0
            out[i, col["pr_rating_deviation"]] = -1.0
            out[i, col["pr_prior_five_share"]] = -1.0
            out[i, col["pr_verified_share"]] = -1.0
            out[i, col["pr_hours_since_prev"]] = -1.0

        _trim(pr.times, t, SEVEN_DAYS)
        p7 = len(pr.times)
        p24 = sum(1 for x in pr.times if t - x <= ONE_DAY)
        out[i, col["pr_count_24h"]] = p24
        out[i, col["pr_count_7d"]] = p7
        # Burst ratio: today's volume against the trailing weekly average.
        weekly_daily_avg = p7 / 7.0
        out[i, col["pr_burst_ratio"]] = (
            p24 / weekly_daily_avg if weekly_daily_avg > 1e-6 else 0.0
        )

        pmax, pmean_sim = _sim_stats(vec, pr.texts)
        out[i, col["pr_peer_similarity_max"]] = pmax
        out[i, col["pr_peer_similarity_mean"]] = pmean_sim

        # Share of this product's trailing-7d reviews written by accounts
        # with essentially no history. A legitimate product attracts a mix of
        # established and new accounts; a bought burst is nearly all fresh
        # ones. Note "fresh account", not "new to this product" -- almost
        # every reviewer is new to a given product, which would make the
        # feature a duplicate of pr_count_7d.
        while pr.new_reviewer_events and t - pr.new_reviewer_events[0][0] > SEVEN_DAYS:
            pr.new_reviewer_events.popleft()
        if pr.new_reviewer_events:
            recent = len(pr.new_reviewer_events)
            fresh = sum(flag for _, flag in pr.new_reviewer_events)
            out[i, col["pr_new_reviewer_share_7d"]] = fresh / recent
        else:
            out[i, col["pr_new_reviewer_share_7d"]] = -1.0

        # ---------------- commit this review into state -------------------
        pr.new_reviewer_events.append((t, int(prior_n <= 1)))

        if rv.first_time is None:
            rv.first_time = t
        rv.n += 1
        rv.rating_sum += ratings[i]
        rv.rating_sq_sum += ratings[i] * ratings[i]
        rv.extreme_count += int(ratings[i] <= 1.5 or ratings[i] >= 4.5)
        rv.len_sum += text_lens[i]
        rv.len_sq_sum += text_lens[i] * text_lens[i]
        rv.times.append(t)
        rv.times_7d.append(t)
        rv.texts.append(vec)
        rv.products[products[i]] = rv.products.get(products[i], 0) + 1
        rv.categories[cats[i]] = rv.categories.get(cats[i], 0) + 1
        rv.verified += int(verified[i])
        if hours[i] < 6 or hours[i] >= 23:
            rv.night_count += 1

        pr.ratings.append(ratings[i])
        pr.times.append(t)
        pr.texts.append(vec)
        pr.verified += int(verified[i])
        pr.five_count += int(ratings[i] >= 4.5)

        if verbose and i and i % 20_000 == 0:
            print(f"  behaviour features: {i:,}/{n:,}")

    return pd.DataFrame(out, columns=BEHAVIOUR_COLUMNS, index=df.index)

