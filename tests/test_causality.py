"""The most important test in the project.

Every behavioural feature is documented as looking strictly backwards in time.
That claim is what justifies computing features over the whole corpus before
splitting, and if it is wrong then every metric this project reports is
inflated. Asserting it in a docstring is worthless; this test proves it.

The proof: recompute the features for a truncated corpus containing only rows
0..i, and check row i's feature vector is bit-identical to row i's vector from
the full corpus. If any feature peeked at row i+1 or later, the two differ.
"""

from __future__ import annotations

import numpy as np
import pytest

from frd.config import SynthConfig
from frd.data.synth import generate
from frd.features.behavior import BEHAVIOUR_COLUMNS, build_behaviour_features


@pytest.fixture(scope="module")
def corpus():
    return generate(SynthConfig(
        n_reviews=1200, n_products=40, n_genuine_reviewers=200,
        n_campaigns=8, seed=11,
    ))


@pytest.fixture(scope="module")
def full_features(corpus):
    return build_behaviour_features(corpus)


@pytest.mark.parametrize("i", [1, 7, 50, 199, 640, 1199])
def test_features_do_not_depend_on_the_future(corpus, full_features, i):
    """Row i's features must be unchanged when all later rows are removed."""
    truncated = corpus.iloc[: i + 1].copy()
    partial = build_behaviour_features(truncated)

    a = full_features.iloc[i].to_numpy(dtype=np.float64)
    b = partial.iloc[i].to_numpy(dtype=np.float64)

    mismatched = [
        BEHAVIOUR_COLUMNS[j]
        for j in range(len(BEHAVIOUR_COLUMNS))
        if not np.isclose(a[j], b[j], rtol=1e-5, atol=1e-6)
    ]
    assert not mismatched, (
        f"row {i}: these features changed when future rows were removed, "
        f"so they leak: {mismatched}"
    )


def test_first_review_of_an_account_has_no_history(corpus, full_features):
    """A reviewer's first-ever review must report zero prior activity."""
    first_idx = corpus.groupby("reviewer_id", observed=True).head(1).index
    sub = full_features.loc[first_idx]
    assert (sub["rv_prior_count"] == 0).all()
    assert (sub["rv_tenure_days"] == 0).all()
    # Sentinel, not 0 -- "unknown" must be distinguishable from "zero".
    assert (sub["rv_hours_since_prev"] == -1.0).all()


def test_prior_count_is_monotone_per_reviewer(corpus, full_features):
    """Prior-count must increase by exactly 1 per review within an account."""
    df = corpus.assign(prior=full_features["rv_prior_count"].to_numpy())
    for _, grp in df.groupby("reviewer_id", observed=True):
        if len(grp) < 2:
            continue
        expected = np.arange(len(grp), dtype=float)
        assert np.array_equal(grp["prior"].to_numpy(), expected)


def test_all_features_finite(full_features):
    arr = full_features.to_numpy(dtype=np.float64)
    assert np.isfinite(arr).all(), "non-finite behavioural feature produced"


def test_burst_features_react_to_a_burst():
    """A synthetic burst must raise the 24h counters."""
    import pandas as pd
    from frd.schema import normalise

    base = pd.Timestamp("2025-01-01", tz="UTC")
    rows = [{
        "review_id": f"R{i}", "product_id": "P1", "reviewer_id": "U1",
        "rating": 5.0, "text": "Great product, highly recommend it.",
        "timestamp": base + pd.Timedelta(minutes=10 * i),
    } for i in range(12)]
    df = normalise(pd.DataFrame(rows))
    f = build_behaviour_features(df)

    assert f["rv_count_24h"].iloc[0] == 0
    assert f["rv_count_24h"].iloc[-1] == 11
    assert f["pr_count_24h"].iloc[-1] == 11
    # Identical text -> near-perfect self-similarity after the first row.
    assert f["rv_self_similarity_max"].iloc[0] == 0.0
    assert f["rv_self_similarity_max"].iloc[-1] > 0.99
