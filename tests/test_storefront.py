"""Storefront tests.

Two things are worth testing here and one thing is not.

WORTH TESTING: that the storefront's arithmetic is the decision policy's
arithmetic. The rating a shopper sees is `mean(rating)` over exactly the
reviews the policy tier `auto_remove` excludes -- no more (which would suppress
unconfirmed reviews) and no fewer (which would show reviews the platform has
already decided to act on). That invariant is checked directly rather than
through a model, so a policy change breaks it loudly.

WORTH TESTING: that the catalogue file and the shop agree. Listings are now
hand-authored data rather than a hash of the product id, so the failure mode
moved: it is no longer "the names change on restart", it is "a listing
references an image that is not in the repo" or "two listings claim the same
slug". Those are checked directly against the file.

WORTH TESTING: that `assign` is stable and neutral. It picks which 61 of the
corpus's products the shop stocks, and if that pick were unstable every URL in
the shop would point at a different product after a restart.

NOT WORTH TESTING: the copy. Asserting a description string would fail on any
edit and prove nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from frd.api import storefront as sf
from frd.config import Config, SynthConfig
from frd.data.catalog import (
    CATALOGUE_PATH, assign, by_slug, departments, listing_for, listings,
    variant_for,
)
from frd.data.splits import temporal_split
from frd.data.synth import generate
from frd.explain import auto_remove_at
from frd.models.pipeline import ALL_BLOCKS, FakeReviewDetector
from frd.train import prepare_dataset

# --- the catalogue ---------------------------------------------------------

def _corpus(n_per_category: int = 20) -> dict[str, tuple[str, int]]:
    """A fake corpus: every department populated, with descending review counts."""
    out: dict[str, tuple[str, int]] = {}
    i = 0
    for dept in departments():
        for k in range(n_per_category):
            out[f"P{i:05d}"] = (dept["key"], 1000 - k)
            i += 1
    return out


def test_every_listing_has_an_image_in_the_repo():
    """A listing whose photograph is missing renders as a broken box in the grid."""
    import json
    from pathlib import Path

    raw = json.loads(CATALOGUE_PATH.read_text())
    web = Path(__file__).resolve().parents[1] / "web" / "public" / "products"
    missing = [p["slug"] for p in raw["products"]
               if not (web / f"{p['slug']}.webp").exists()]
    assert not missing, f"listings with no image: {missing}"


def test_listings_are_internally_consistent():
    items = listings()
    assert len(items) >= 40, "the shop should look stocked"

    slugs = [i["slug"] for i in items]
    assert len(set(slugs)) == len(slugs), "duplicate slug"
    skus = [i["sku"] for i in items]
    assert len(set(skus)) == len(skus), "duplicate SKU"

    known = {d["key"] for d in departments()}
    for item in items:
        assert item["category"] in known, f"{item['slug']} is in no department"
        assert item["price"] > 0 and item["name"] and item["brand"]
        assert item["summary"] and item["description"]
        # Variant stock must partition the product's stock, not multiply it.
        assert sum(v.stock for v in item["variants"]) == item["stock"]
        assert len({v.sku for v in item["variants"]}) == len(item["variants"])


def test_assign_is_stable_and_slug_addressable():
    """The same corpus must produce the same shop, or every URL breaks on restart."""
    corpus = _corpus()
    a, b = assign(corpus), assign(dict(reversed(list(corpus.items()))))
    assert {k: v.slug for k, v in a.items()} == {k: v.slug for k, v in b.items()}

    # One listing per corpus product and one corpus product per listing.
    assert len({p.slug for p in a.values()}) == len(a)
    assert len(by_slug(a)) == len(a)
    assert all(pid == p.product_id for pid, p in a.items())


def test_assign_ranks_by_review_count_not_by_anything_scored():
    """The busiest corpus product in a department gets that department's first listing.

    The rule has to be neutral with respect to the model: ranking by
    manipulation would stock the shop with the products that best demonstrate
    the detector and make every headline number a selection effect.
    """
    corpus = {"P1": ("toys", 5), "P2": ("toys", 900), "P3": ("toys", 50)}
    got = assign(corpus)
    first = next(i for i in listings() if i["category"] == "toys")
    assert got["P2"].slug == first["slug"]


def test_assign_skips_departments_the_corpus_cannot_fill():
    """A shop cannot list a product with no reviews behind it."""
    got = assign({"P1": ("toys", 100)})
    assert len(got) == 1
    assert got["P1"].category == "toys"


def test_variant_lookup_is_the_pricing_path():
    """Carts price from here, and it must not need a model or a corpus."""
    item = next(i for i in listings() if len(i["variants"]) > 1)
    found = variant_for(item["slug"], item["variants"][0].sku)
    assert found is not None
    listing, variant = found
    assert listing["slug"] == item["slug"]
    assert variant.price == item["price"]
    assert variant_for(item["slug"], "NO-SUCH-SKU") is None
    assert listing_for("no-such-product") is None


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
    assert stats["counts"]["n_listed"] > 0
    # The shop is a curated subset; conflating the two counts would let the
    # home page imply it had checked every product in the corpus.
    assert stats["counts"]["n_corpus_products"] >= stats["counts"]["n_listed"]
    # The policy constants are served, not duplicated in the client.
    assert stats["material_delta"] == sf.MATERIAL_DELTA
    assert stats["auto_remove_at"] == pytest.approx(auto_remove_at(stats["threshold"]))

    page = client.get("/shop/products", params={"limit": 5}).json()
    assert page["total"] == stats["counts"]["n_listed"]
    assert len(page["items"]) == 5
    for item in page["items"]:
        assert item["name"] and item["price"] > 0
        assert item["n_shown"] + item["n_withheld"] == item["n_reviews"]


def test_product_detail_partitions_every_review(client):
    """Nothing may be dropped: shown + withheld must account for the corpus."""
    slug = client.get("/shop/products", params={"limit": 1}).json()["items"][0]["slug"]
    d = client.get(f"/shop/products/{slug}", params={"limit": 100}).json()
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
    ).json()["items"][0]["slug"]
    d = client.get(f"/shop/products/{pid}", params={"limit": 5}).json()
    review = (d["reviews"] or d["withheld"])[0]

    console = client.get(f"/queue/{review['review_id']}").json()
    assert console["score"] == pytest.approx(review["score"], abs=1e-9)
    assert console["decision"] == review["decision"]


def test_withheld_reviews_carry_reason_codes(client):
    """A withheld review with no stated reason is an unauditable accusation."""
    stats = client.get("/shop/stats").json()
    for mover in stats["movers"]:
        d = client.get(f"/shop/products/{mover['slug']}").json()
        if d["withheld"]:
            assert d["withheld"][0]["reasons"], "withheld review has no reasons"
            return
    pytest.skip("no withheld reviews in this small corpus")


def test_unknown_product_is_404(client):
    assert client.get("/shop/products/NOT_A_PRODUCT").status_code == 404
