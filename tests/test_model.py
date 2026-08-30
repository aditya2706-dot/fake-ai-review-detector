import numpy as np
import pandas as pd
import pytest

from frd.config import Config, SynthConfig
from frd.data.synth import generate
from frd.data.splits import cold_reviewer_split, temporal_split
from frd.explain import explain
from frd.models.pipeline import ALL_BLOCKS, FakeReviewDetector
from frd.schema import LABEL
from frd.train import prepare_dataset


@pytest.fixture(scope="module")
def cfg():
    c = Config()
    # Keep the fitted model small so the suite stays fast.
    c.text.word_max_features = 3_000
    c.text.char_max_features = 2_000
    c.model.lgbm_params["n_estimators"] = 60
    return c


@pytest.fixture(scope="module")
def prepared():
    df = generate(SynthConfig(
        n_reviews=2500, n_products=60, n_genuine_reviewers=400,
        n_campaigns=12, seed=5,
    ))
    return prepare_dataset(df, verbose=False)


@pytest.fixture(scope="module")
def split(prepared):
    return temporal_split(prepared)


@pytest.fixture(scope="module")
def fitted(split, cfg):
    det = FakeReviewDetector(blocks=ALL_BLOCKS, cfg=cfg, oof_folds=3)
    det.fit(split.train)
    det.calibrate(split.val)
    return det


def test_probabilities_are_in_range(fitted, split):
    p = fitted.predict_proba(split.test)
    assert len(p) == len(split.test)
    assert (p >= 0).all() and (p <= 1).all()


def test_beats_random_ranking(fitted, split):
    from sklearn.metrics import average_precision_score
    y = split.test[LABEL].to_numpy(int)
    ap = average_precision_score(y, fitted.predict_proba(split.test))
    # A weak but non-trivial bar: comfortably above the base rate.
    assert ap > y.mean() * 1.5, f"PR-AUC {ap:.3f} vs base rate {y.mean():.3f}"


@pytest.mark.parametrize("blocks", [
    ("text",), ("linguistic",), ("behaviour",), ("text", "behaviour"),
])
def test_every_ablation_fits_and_scores(split, cfg, blocks):
    det = FakeReviewDetector(blocks=blocks, cfg=cfg, oof_folds=2)
    det.fit(split.train)
    p = det.predict_proba(split.test)
    assert np.isfinite(p).all()


def test_unknown_block_rejected(cfg):
    with pytest.raises(ValueError, match="unknown block"):
        FakeReviewDetector(blocks=("text", "astrology"), cfg=cfg)


def test_empty_blocks_rejected(cfg):
    with pytest.raises(ValueError, match="at least one"):
        FakeReviewDetector(blocks=(), cfg=cfg)


def test_single_class_training_rejected(split, cfg):
    only_genuine = split.train[split.train[LABEL] == 0]
    with pytest.raises(ValueError, match="single class"):
        FakeReviewDetector(blocks=("linguistic",), cfg=cfg).fit(only_genuine)


def test_predict_before_fit_raises(cfg, split):
    with pytest.raises(RuntimeError, match="not fitted"):
        FakeReviewDetector(blocks=("linguistic",), cfg=cfg).predict_proba(split.test)


def test_calibration_preserves_ranking(fitted, split):
    """Calibration is monotone, so PR-AUC must not move."""
    from sklearn.metrics import average_precision_score
    y = split.test[LABEL].to_numpy(int)
    raw = fitted.decision_score(split.test)
    cal = fitted.predict_proba(split.test)
    assert np.isclose(
        average_precision_score(y, raw), average_precision_score(y, cal),
        atol=1e-9,
    )


def test_calibration_improves_probability_accuracy(split, cfg):
    """Reweighted raw scores are biased; calibration should reduce the gap."""
    det = FakeReviewDetector(blocks=("linguistic", "behaviour"), cfg=cfg)
    det.fit(split.train)
    y = split.test[LABEL].to_numpy(int)
    raw_gap = abs(det.decision_score(split.test).mean() - y.mean())
    det.calibrate(split.val)
    cal_gap = abs(det.predict_proba(split.test).mean() - y.mean())
    assert cal_gap <= raw_gap + 0.02


def test_save_load_roundtrip_is_score_identical(fitted, split, tmp_path):
    p = fitted.save(tmp_path / "m.joblib")
    assert p.exists() and p.with_suffix(".meta.json").exists()
    loaded = FakeReviewDetector.load(p)
    np.testing.assert_allclose(
        fitted.predict_proba(split.test), loaded.predict_proba(split.test),
        rtol=1e-6,
    )
    assert loaded.threshold_ == fitted.threshold_


def test_threshold_shifts_flag_volume(fitted, split):
    low = fitted.predict(split.test, threshold=0.2).sum()
    high = fitted.predict(split.test, threshold=0.8).sum()
    assert low >= high


def test_explanations_are_produced(fitted, split):
    exps = explain(fitted, split.test.head(5))
    assert len(exps) == 5
    for e in exps:
        assert 0 <= e.score <= 1
        assert e.decision in {"allow", "queue_for_review", "auto_remove"}
        assert e.feature_contributions or e.method.startswith("unavailable")


def test_top_flagged_reviews_have_reasons(fitted, split):
    """The highest-scoring reviews must come with human-readable reasons."""
    s = fitted.predict_proba(split.test)
    top = split.test.iloc[np.argsort(-s)[:5]]
    exps = explain(fitted, top)
    assert any(e.reasons for e in exps), "no reason codes on the top-scored reviews"


def test_cold_reviewer_split_shares_no_accounts(prepared):
    sp = cold_reviewer_split(prepared)
    tr = set(sp.train.reviewer_id.astype(str))
    te = set(sp.test.reviewer_id.astype(str))
    assert not (tr & te)


def test_scoring_a_bare_review_without_prepared_columns(fitted):
    """The API path: raw review in, no precomputed feature columns."""
    from frd.schema import normalise
    raw = normalise(pd.DataFrame([{
        "review_id": "X1", "product_id": "P1", "reviewer_id": "U1",
        "rating": 5.0, "text": "Best purchase ever! Highly recommend!",
        "timestamp": pd.Timestamp("2025-06-01", tz="UTC"),
    }]))
    p = fitted.predict_proba(raw)
    assert p.shape == (1,) and 0 <= p[0] <= 1


def test_reason_guards_suppress_misleading_claims():
    """A reason must not fire when its value contradicts the sentence.

    A 6% text similarity can still contribute positively to the score, but
    "Text is 6% similar to another review" reads as incriminating evidence in a
    moderation UI when it is nothing of the sort.
    """
    from frd.explain import _reason_allowed

    assert not _reason_allowed("rv_self_similarity_max", 0.06)
    assert _reason_allowed("rv_self_similarity_max", 0.93)

    # "Only X% of past reviews were verified" only makes sense when X is low.
    assert not _reason_allowed("rv_verified_share", 0.85)
    assert _reason_allowed("rv_verified_share", 0.05)

    # A -1 sentinel means "unknown" and must never become a claim.
    assert not _reason_allowed("rv_verified_share", -1.0)

    # Burst language needs an actual burst.
    assert not _reason_allowed("rv_count_24h", 1)
    assert _reason_allowed("rv_count_24h", 14)

    # Ungated features pass through.
    assert _reason_allowed("text_score", 0.5)


def test_every_reason_template_formats(fitted):
    """Every template must render without raising on a plausible value."""
    from frd.explain import REASON_TEMPLATES
    for name, tpl in REASON_TEMPLATES.items():
        rendered = tpl.format(v=0.5)
        assert rendered and "{" not in rendered, name


def test_genuine_first_time_reviewer_is_not_flagged(fitted):
    """The false-positive property that matters most.

    With no history every behavioural feature is a "no history" sentinel, and
    in training that pattern correlates with fraud (farm accounts posting their
    first review). The risk is that the model then flags every genuine
    first-time reviewer, which would be a serious product failure.

    A specific, measured, verified first review must stay well below threshold.
    """
    from frd.schema import normalise

    raw = normalise(pd.DataFrame([{
        "review_id": "FIRST-1", "product_id": "P1", "reviewer_id": "BRAND-NEW",
        "rating": 4.0, "title": "Does the job",
        "text": ("The strap measured 31cm not the 29cm listed and it ran about "
                 "six hours before needing a charge. Fine for the price though."),
        "timestamp": pd.Timestamp("2025-06-01", tz="UTC"),
        "verified_purchase": True, "helpful_votes": 2, "category": "electronics",
    }]))
    score = float(fitted.predict_proba(raw)[0])
    assert score < 0.5, (
        f"a genuine, specific, verified first review scored {score:.3f}; "
        "the model is treating 'no history' as sufficient evidence of fraud"
    )


def test_promotional_text_outscores_specific_text(fitted):
    """The core linguistic hypothesis, end to end through the model."""
    from frd.schema import normalise

    def build(rid, text):
        return {
            "review_id": rid, "product_id": "P1", "reviewer_id": "U9",
            "rating": 5.0, "title": "Great", "text": text,
            "timestamp": pd.Timestamp("2025-06-01", tz="UTC"),
            "verified_purchase": False, "helpful_votes": 0,
            "category": "electronics",
        }

    df = normalise(pd.DataFrame([
        build("PROMO", "Best purchase ever! Really amazing build quality. "
                       "Super excellent packaging. Highly recommend to everyone!"),
        build("SPECIFIC", "Arrived two days early in a battered box. Measured "
                          "31cm not the 29cm listed, and the manual is a single "
                          "badly translated sheet."),
    ]))
    scores = dict(zip(df["review_id"].astype(str), fitted.predict_proba(df)))
    assert scores["PROMO"] > scores["SPECIFIC"], scores
