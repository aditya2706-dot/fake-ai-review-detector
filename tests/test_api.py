"""API tests, including the train/serve consistency check.

The last test here is the one that matters: scoring a review through the HTTP
path must give the same number as scoring it offline. If the API's feature
assembly ever drifts from `prepare_dataset`, that test fails -- which is the
only cheap way to catch training/serving skew before production does.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from frd.config import Config, SynthConfig
from frd.data.splits import temporal_split
from frd.data.synth import generate
from frd.models.pipeline import ALL_BLOCKS, FakeReviewDetector
from frd.train import prepare_dataset


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
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
    # ALL_BLOCKS, deliberately. The graph block is the one most likely to skew
    # between training and serving: it is the only family whose values depend on
    # accounts other than the author's, so the API's assembled context frame has
    # the furthest to fall short of what `prepare_dataset` saw. Testing the skew
    # guarantee against the three easy blocks would have been testing the wrong
    # three.
    det = FakeReviewDetector(blocks=ALL_BLOCKS, cfg=cfg, oof_folds=2)
    det.fit(split.train)
    det.calibrate(split.val)
    det.set_threshold(0.5)

    path = tmp_path_factory.mktemp("model") / "detector.joblib"
    det.save(path)
    return det, path, df


@pytest.fixture(scope="module")
def client(trained, monkeypatch_module=None):
    _, path, _ = trained
    from frd.api import main as api_main
    api_main.MODEL_PATH = path
    with TestClient(api_main.app) as c:
        yield c


def _review(**over):
    base = {
        "review_id": "R-TEST-1", "product_id": "P00001",
        "reviewer_id": "U000001", "rating": 5.0,
        "text": "Best purchase ever! Amazing quality! Highly recommend!",
        "timestamp": "2025-06-01T12:00:00Z", "title": "Great!",
        "verified_purchase": False, "helpful_votes": 0, "category": "electronics",
    }
    base.update(over)
    return base


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert set(body["blocks"]) == set(ALL_BLOCKS)
    assert body["calibration"] == "sigmoid"


def test_score_without_context(client):
    r = client.post("/score", json={"review": _review(), "explain": True})
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["score"] <= 1.0
    assert b["decision"] in {"allow", "queue_for_review", "auto_remove"}
    assert b["behavioural_context_used"] is False
    assert b["n_context_reviews"] == 0


def test_score_with_context_uses_it(client):
    ctx = [{
        "review_id": f"C{i}", "product_id": "P00001",
        "reviewer_id": "U000001", "rating": 5.0,
        "text": "Amazing product! Highly recommend to everyone!",
        "timestamp": f"2025-05-31T{10 + i:02d}:00:00Z",
        "verified_purchase": False, "category": "electronics",
    } for i in range(8)]
    r = client.post("/score", json={"review": _review(), "context": ctx,
                                    "explain": True})
    assert r.status_code == 200
    b = r.json()
    assert b["behavioural_context_used"] is True
    assert b["n_context_reviews"] == 8
    # A burst of near-identical reviews by the same author should surface a
    # reason mentioning volume or similarity.
    joined = " ".join(b["reasons"]).lower()
    assert b["reasons"], "context supplied but no reason codes produced"
    assert any(k in joined for k in ("review", "similar", "hour", "day"))


def test_future_context_is_discarded(client):
    """Context at/after the target timestamp must be dropped, not used."""
    future = [{
        "review_id": "FUT1", "product_id": "P00001", "reviewer_id": "U000001",
        "rating": 5.0, "text": "later review", "category": "electronics",
        "timestamp": "2025-07-01T00:00:00Z",   # after the target
    }]
    r = client.post("/score", json={"review": _review(), "context": future})
    assert r.status_code == 200
    assert r.json()["n_context_reviews"] == 0


def test_batch_scoring(client):
    reviews = [_review(review_id=f"B{i}", text=t) for i, t in enumerate([
        "Amazing! Perfect! Best ever! Highly recommend!",
        "Arrived Tuesday, the 31cm strap was 2cm short but it works fine.",
        "ok",
    ])]
    r = client.post("/score/batch", json={"reviews": reviews, "explain": False})
    assert r.status_code == 200
    b = r.json()
    assert b["count"] == 3 and len(b["results"]) == 3
    assert all(0 <= x["score"] <= 1 for x in b["results"])


def test_invalid_rating_rejected(client):
    r = client.post("/score", json={"review": _review(rating=11)})
    assert r.status_code == 422


def test_model_info_endpoint(client):
    b = client.get("/model").json()
    assert b["n_stage_b_features"] > 0
    assert b["behaviour_features"] > 0
    assert isinstance(b["top_features"], list)


def test_missing_model_returns_503(trained):
    from frd.api import main as api_main
    original = api_main.STATE.get("detector")
    api_main.STATE["detector"] = None
    try:
        with TestClient(api_main.app) as c:
            api_main.STATE["detector"] = None   # lifespan may reload
            r = c.post("/score", json={"review": _review()})
            assert r.status_code == 503
    finally:
        api_main.STATE["detector"] = original


def test_api_and_offline_scores_agree(trained, client):
    """No training/serving skew: same review, same number, both paths."""
    det, _, _ = trained
    review = _review(review_id="SKEW-1")

    api_score = client.post(
        "/score", json={"review": review, "explain": False}
    ).json()["score"]

    offline = prepare_dataset(
        pd.DataFrame([{**review, "timestamp": pd.Timestamp(review["timestamp"])}]),
        verbose=False,
    )
    offline_score = float(det.predict_proba(offline)[0])

    assert abs(api_score - offline_score) < 1e-6, (
        f"API {api_score:.6f} != offline {offline_score:.6f}: "
        "the serving feature path has drifted from training"
    )
