"""Proofs for the co-review graph block.

The graph features carry the same causality obligation as the behavioural ones
and a sharper version of the same risk: a graph statistic is computed from
edges contributed by *other* accounts, so "did this row see the future" is much
harder to eyeball than it is for a running per-reviewer counter. Hence the same
proof, repeated here rather than assumed: recompute on a corpus truncated at
row i and demand row i is bit-identical.

The structural tests below are the other half. Causality says the features do
not cheat; they do not say the features measure anything. A planted farm and a
planted crowd of strangers, run through the same function, must come out
looking different -- otherwise the block is honest and useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frd.config import SynthConfig
from frd.data.synth import generate
from frd.features.graph import GRAPH_COLUMNS, build_graph_features
from frd.schema import normalise


@pytest.fixture(scope="module")
def corpus():
    return generate(SynthConfig(
        n_reviews=1200, n_products=40, n_genuine_reviewers=200,
        n_campaigns=8, seed=11,
    ))


@pytest.fixture(scope="module")
def full_features(corpus):
    return build_graph_features(corpus)


@pytest.mark.parametrize("i", [1, 7, 50, 199, 640, 1199])
def test_graph_features_do_not_depend_on_the_future(corpus, full_features, i):
    """Row i's features must be unchanged when all later rows are removed."""
    truncated = corpus.iloc[: i + 1].copy()
    partial = build_graph_features(truncated)

    a = full_features.iloc[i].to_numpy(dtype=np.float64)
    b = partial.iloc[i].to_numpy(dtype=np.float64)

    mismatched = [
        GRAPH_COLUMNS[j]
        for j in range(len(GRAPH_COLUMNS))
        if not np.isclose(a[j], b[j], rtol=1e-5, atol=1e-6)
    ]
    assert not mismatched, (
        f"row {i}: these graph features changed when future rows were "
        f"removed, so they leak: {mismatched}"
    )


def test_first_review_of_an_account_has_no_neighbourhood(corpus, full_features):
    """An account's first-ever review must report an empty neighbourhood."""
    first_idx = corpus.groupby("reviewer_id", observed=True).head(1).index
    sub = full_features.loc[first_idx]
    assert (sub["gp_partner_count"] == 0).all()
    assert (sub["gp_component_size"] == 1).all()
    # Sentinel, not 0 -- "no partners at all" must stay distinguishable from
    # "many partners, no overlap", which a 0-fill would merge.
    assert (sub["gp_max_shared_products"] == -1.0).all()
    assert (sub["gp_max_partner_jaccard"] == -1.0).all()
    assert (sub["gp_partner_concentration"] == -1.0).all()


def test_all_features_finite(full_features):
    arr = full_features.to_numpy(dtype=np.float64)
    assert np.isfinite(arr).all(), "non-finite graph feature produced"


# --------------------------------------------------------------------------
# Structural tests: does the block actually separate a farm from a crowd?
# --------------------------------------------------------------------------

def _frame(rows: list[dict]) -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01", tz="UTC")
    return normalise(pd.DataFrame([{
        "review_id": f"R{i}",
        "product_id": r["p"],
        "reviewer_id": r["u"],
        "rating": 5.0,
        "text": "Great product, works well.",
        "timestamp": base + pd.Timedelta(hours=r["h"]),
    } for i, r in enumerate(rows)]))


def test_a_ring_of_accounts_working_the_same_products_lights_up():
    """Six accounts hitting the same five products is the farm signature."""
    workers = [f"F{i}" for i in range(6)]
    rows = []
    h = 0
    for prod in [f"P{j}" for j in range(5)]:
        for w in workers:
            rows.append({"u": w, "p": prod, "h": h})
            h += 1
    f = build_graph_features(_frame(rows))

    last = f.iloc[-1]
    # By the fifth product every worker has overlapped with every other on
    # four previous products.
    assert last["gp_partner_count"] == 5
    assert last["gp_max_shared_products"] == 4
    assert last["gp_strong_partner_count"] == 5
    # Nearly everything each of them has touched, the others touched too.
    assert last["gp_max_partner_jaccard"] > 0.7
    # All six are one strong-tie component.
    assert last["gp_component_size"] == 6
    # And the product's cohort is a clique: every pair has met before.
    assert last["gp_cohort_strong_pair_density"] == 1.0


def test_strangers_on_a_popular_product_stay_quiet():
    """Distinct accounts sharing one product must not look coordinated.

    This is the false-positive case that matters: any popular product puts
    hundreds of unrelated buyers on the same node, and a naive co-review
    feature would call all of them associates.
    """
    rows = [{"u": f"U{i}", "p": "P0", "h": i} for i in range(30)]
    f = build_graph_features(_frame(rows))

    # Nobody is ever promoted into a component -- including the pair who were
    # the product's only two reviewers at the time, whose single overlap the
    # surprise weight alone would have scored highly enough to promote.
    assert f["gp_component_size"].max() == 1
    assert f["gp_strong_partner_count"].max() == 0
    # Every cohort these accounts arrive into is a room full of strangers.
    density = f["gp_cohort_strong_pair_density"]
    assert density[density >= 0].max() == 0.0


def test_overlap_counts_products_not_reviews():
    """Two accounts hammering ONE product are not a five-product ring."""
    rows = []
    for h in range(5):
        rows.append({"u": "A", "p": "P0", "h": h * 2})
        rows.append({"u": "B", "p": "P0", "h": h * 2 + 1})
    f = build_graph_features(_frame(rows))

    # A and B overlap on exactly one product no matter how often they post.
    assert f["gp_max_shared_products"].iloc[-1] == 1
    assert f["gp_component_size"].iloc[-1] == 1


def test_arriving_together_repeatedly_is_counted():
    """Accounts that keep landing in the same 48h window on a product."""
    rows = []
    for j, prod in enumerate(["P0", "P1", "P2"]):
        rows.append({"u": "A", "p": prod, "h": j * 24 * 30})
        rows.append({"u": "B", "p": prod, "h": j * 24 * 30 + 1})
    f = build_graph_features(_frame(rows))

    assert f["gp_tight_cooccur_max"].iloc[-1] == 2
    assert f["gp_tight_partner_count"].iloc[-1] == 1


def test_a_late_arrival_sees_the_cohort_that_preceded_it():
    """The tie-to-product features must fire for the account joining last."""
    workers = [f"F{i}" for i in range(4)]
    rows = []
    h = 0
    for prod in ["P0", "P1"]:            # build the ring
        for w in workers:
            rows.append({"u": w, "p": prod, "h": h})
            h += 1
    for w in workers[:3]:                 # three of them seed a new product
        rows.append({"u": w, "p": "P2", "h": h})
        h += 1
    rows.append({"u": workers[3], "p": "P2", "h": h})   # the fourth follows

    f = build_graph_features(_frame(rows))
    last = f.iloc[-1]
    assert last["gp_tie_to_product_share"] == 1.0
    assert last["gp_tie_to_product_surprise"] > 0.0
