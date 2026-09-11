"""Storefront tests.

Two things are worth testing here and one thing is not.

WORTH TESTING: that the storefront's arithmetic is the decision policy's
arithmetic. The rating a shopper sees is `mean(rating)` over exactly the
reviews the policy tier `auto_remove` excludes -- no more (which would suppress
unconfirmed reviews) and no fewer (which would show reviews the platform has
already decided to act on). That invariant is checked directly rather than
through a model, so a policy change breaks it loudly.

WORTH TESTING: that the catalogue is a function, not a lottery. Product names
are derived from ids at request time, so a non-deterministic derivation would
rename the whole shop on restart and make every screenshot, bookmark and bug
report worthless.

NOT WORTH TESTING: the specific names. `product_for("P1", "toys").name ==
"Cogsworth One Board Game"` would fail on any vocabulary edit and prove
nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from frd.api import storefront as sf
from frd.config import Config, SynthConfig
from frd.data.catalog import catalog_for, product_for
from frd.data.splits import temporal_split
from frd.data.synth import generate
from frd.explain import auto_remove_at
from frd.models.pipeline import ALL_BLOCKS, FakeReviewDetector
from frd.train import prepare_dataset

# --- the catalogue ---------------------------------------------------------

def test_catalog_is_deterministic():
    """Same id in, same product out. Twice, from scratch."""
    a = product_for("P00042", "electronics")
    b = product_for("P00042", "electronics")
    assert a == b
    assert a.name and a.brand and a.price > 0


def test_catalog_is_stable_across_processes():
    """The seed must not come from `hash()`, which Python salts per process.

    Asserted against a value captured from a separate interpreter rather than
    against a literal name, so the check survives a vocabulary edit but still
    fails if the hash is swapped for a salted one.
    """
    import subprocess
    import sys

    code = (
        "from frd.data.catalog import product_for;"
        "p = product_for('P00042', 'electronics');"
        "print(p.name)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == product_for("P00042", "electronics").name


def test_catalog_ids_are_distinguishable():
    """Different products must not present as the same listing."""
    ids = [f"P{i:05d}" for i in range(400)]
    cats = ["electronics", "toys", "beauty", "office"] * 100
    cat = catalog_for(ids, cats)
    names = [p.name for p in cat.values()]
    assert len(names) == len(set(names)), "two listings share a name"


def test_catalog_covers_unknown_categories():
    """A real corpus will contain a category this vocabulary has never seen."""
    p = product_for("P00001", "farm_equipment")
    assert p.name and p.category == "farm_equipment"


# --- the rating arithmetic -------------------------------------------------

def _store(rows: list[tuple[str, float, float]]) -> dict:
    """A minimal scored corpus: (product_id, rating, score) per review."""
    df = pd.DataFrame({
        "review_id": [f"R{i}" for i in range(len(rows))],
        "product_id": [r[0] for r in rows],
        "category": ["toys"] * len(rows),
        "rating": [r[1] for r in rows],
        "score": [r[2] for r in rows],
        "timestamp": pd.date_range("2024-01-01", periods=len(rows), tz="UTC"),
    })
    return {"frame": df, "train_end": df["timestamp"].max()}


def test_adjusted_rating_excludes_exactly_the_auto_remove_tier():
    """The invariant the whole storefront rests on."""
    thr = 0.30
    remove_at = auto_remove_at(thr)          # 0.55
    rows = [
        ("A", 5.0, 0.90),                    # auto_remove  -> withheld
        ("A", 5.0, remove_at),               # exactly at the line -> withheld
        ("A", 1.0, 0.40),                    # queue_for_review -> shown
        ("A", 3.0, 0.01),                    # allow -> shown
    ]
    agg = sf._aggregate(_store(rows), thr)
    got = agg.loc["A"]

    assert got["n_withheld"] == 2
    assert got["n_under_review"] == 1
    assert got["n_shown"] == 2
    assert got["raw_rating"] == pytest.approx(3.5)          # (5+5+1+3)/4
    # Mean of the shown ratings only -- the queued review counts, because a
    # model's suspicion is not grounds for removing it from the rating.
    assert got["adjusted_rating"] == pytest.approx(2.0)     # (1+3)/2


def test_direction_and_materiality():
    """Cleaning runs both ways, and small moves are not announced."""
    thr = 0.30
    inflated = sf._aggregate(_store([
        ("A", 5.0, 0.99), ("A", 5.0, 0.99), ("A", 2.0, 0.01),
    ]), thr).loc["A"]
    assert inflated["direction"] == "inflated"
    assert inflated["delta"] < 0

    attacked = sf._aggregate(_store([
        ("B", 1.0, 0.99), ("B", 1.0, 0.99), ("B", 5.0, 0.01),
    ]), thr).loc["B"]
    assert attacked["direction"] == "attacked"
    assert attacked["delta"] > 0

    # A withheld review that does not move the mean is not a story.
    flat = sf._aggregate(_store([
        ("C", 4.0, 0.99), ("C", 4.0, 0.01), ("C", 4.0, 0.01),
    ]), thr).loc["C"]
    assert flat["n_withheld"] == 1
    assert flat["material"] is np.False_ or not flat["material"]
    assert flat["direction"] == "unchanged"


def test_fully_withheld_product_has_no_cleaned_rating():
    """Null, not the raw rating: 'unchanged' there would be a lie."""
    agg = sf._aggregate(_store([("A", 5.0, 0.99), ("A", 5.0, 0.99)]), 0.30)
    got = agg.loc["A"]
    assert got["n_shown"] == 0
    assert pd.isna(got["adjusted_rating"])
    assert got["adjusted_thin"]


def test_aggregate_cache_is_scoped_to_its_corpus():
    """Two corpora must never be served each other's aggregate.

    This held a bug: the cache was a module global keyed on `id(frame)`, and
    CPython hands out a collected object's id again, so a freshly built frame
    could be answered from the previous one's entry. The cache now lives in the
    corpus store, which cannot outlive the frame it describes.
    """
    import gc

    first = _store([("A", 5.0, 0.99), ("A", 1.0, 0.01)])
    agg_a = sf._aggregate(first, 0.30)
    assert list(agg_a.index) == ["A"]
    assert "agg_cache" in first, "the cache must live in the store, not module scope"

    del first, agg_a
    gc.collect()                       # frees the frame whose id may be reused

    second = _store([("B", 2.0, 0.01), ("B", 4.0, 0.01)])
    agg_b = sf._aggregate(second, 0.30)
    assert list(agg_b.index) == ["B"]
    assert agg_b.loc["B"]["raw_rating"] == pytest.approx(3.0)


def test_thin_flag_tracks_survivor_count():
    thr = 0.30
    many = [("A", 4.0, 0.01)] * (sf.MIN_SHOWN_FOR_CONFIDENCE + 1)
    assert not sf._aggregate(_store(many), thr).loc["A"]["adjusted_thin"]
    few = [("B", 4.0, 0.01)] * (sf.MIN_SHOWN_FOR_CONFIDENCE - 1)
    assert sf._aggregate(_store(few), thr).loc["B"]["adjusted_thin"]


# --- the served surface ----------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A real model over a small corpus, served over HTTP."""
    cfg = Config()
    cfg.text.word_max_features = 2_000
    cfg.text.char_max_features = 1_000
    cfg.model.lgbm_params["n_estimators"] = 50

    df = generate(SynthConfig(
        n_reviews=2000, n_products=50, n_genuine_reviewers=300,
        n_campaigns=10, seed=3,
    ))
    prepared = prepare_dataset(df, verbose=False)
    split = temporal_split(prepared)
    det = FakeReviewDetector(blocks=ALL_BLOCKS, cfg=cfg, oof_folds=2)
    det.fit(split.train)
    det.calibrate(split.val)
    det.set_threshold(0.5)

    model_path = tmp_path_factory.mktemp("model") / "detector.joblib"
    det.save(model_path)
    corpus_path = tmp_path_factory.mktemp("data") / "reviews.csv"
    df.to_csv(corpus_path, index=False)

    from frd.api import main as api_main
    from frd.api import queue as api_queue

    api_main.MODEL_PATH = model_path
    api_queue._CACHE.clear()

    import os
    os.environ["FRD_DATA_PATH"] = str(corpus_path)
    try:
        with TestClient(api_main.app) as c:
            yield c
    finally:
        os.environ.pop("FRD_DATA_PATH", None)
        api_queue._CACHE.clear()


def test_stats_and_products_are_consistent(client):
    stats = client.get("/shop/stats").json()
    assert stats["counts"]["n_products"] > 0
    # The policy constants are served, not duplicated in the client.
    assert stats["material_delta"] == sf.MATERIAL_DELTA
    assert stats["auto_remove_at"] == pytest.approx(auto_remove_at(stats["threshold"]))

    page = client.get("/shop/products", params={"limit": 5}).json()
    assert page["total"] == stats["counts"]["n_products"]
    assert len(page["items"]) == 5
    for item in page["items"]:
        assert item["name"] and item["price"] > 0
        assert item["n_shown"] + item["n_withheld"] == item["n_reviews"]


def test_product_detail_partitions_every_review(client):
    """Nothing may be dropped: shown + withheld must account for the corpus."""
    pid = client.get("/shop/products", params={"limit": 1}).json()["items"][0]["product_id"]
    d = client.get(f"/shop/products/{pid}", params={"limit": 100}).json()
    assert d["total_shown"] + d["total_withheld"] == d["product"]["n_reviews"]
    assert sum(d["rating_histogram"]["raw"].values()) == d["product"]["n_reviews"]

    # Every shown review is below the withholding line; every withheld one is
    # at or above it. This is the claim the page makes to a shopper.
    for r in d["reviews"]:
        assert r["score"] < d["auto_remove_at"]
        assert r["decision"] in ("allow", "queue_for_review")
    for r in d["withheld"]:
        assert r["score"] >= d["auto_remove_at"]
        assert r["decision"] == "auto_remove"


def test_storefront_and_console_agree_on_a_review(client):
    """The two surfaces must not disagree about the same review.

    Same discipline as `test_api.py::test_api_and_offline_scores_agree`: the
    storefront reads the console's corpus cache rather than scoring anything
    itself, and this is what would fail if that ever changed.
    """
    pid = client.get(
        "/shop/products", params={"limit": 1, "sort": "reviews"},
    ).json()["items"][0]["product_id"]
    d = client.get(f"/shop/products/{pid}", params={"limit": 5}).json()
    review = (d["reviews"] or d["withheld"])[0]

    console = client.get(f"/queue/{review['review_id']}").json()
    assert console["score"] == pytest.approx(review["score"], abs=1e-9)
    assert console["decision"] == review["decision"]


def test_withheld_reviews_carry_reason_codes(client):
    """A withheld review with no stated reason is an unauditable accusation."""
    stats = client.get("/shop/stats").json()
    for mover in stats["movers"]:
        d = client.get(f"/shop/products/{mover['product_id']}").json()
        if d["withheld"]:
            assert d["withheld"][0]["reasons"], "withheld review has no reasons"
            return
    pytest.skip("no withheld reviews in this small corpus")


def test_unknown_product_is_404(client):
    assert client.get("/shop/products/NOT_A_PRODUCT").status_code == 404
