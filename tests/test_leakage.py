"""Guards against the failure modes that silently inflate results.

These are negative-control tests: they check the pipeline CANNOT find signal
where none exists. A pipeline that scores well on shuffled labels has a bug
(usually an identifier leaking in, or a target-derived feature), and without a
test like this the bug reads as a great result.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from frd.config import Config, SynthConfig
from frd.data.splits import temporal_split
from frd.data.synth import generate
from frd.models.pipeline import FakeReviewDetector
from frd.schema import LABEL
from frd.train import prepare_dataset


@pytest.fixture(scope="module")
def cfg():
    c = Config()
    c.text.word_max_features = 2_000
    c.text.char_max_features = 1_000
    c.model.lgbm_params["n_estimators"] = 50
    return c


@pytest.fixture(scope="module")
def prepared():
    df = generate(SynthConfig(
        n_reviews=2500, n_products=60, n_genuine_reviewers=400,
        n_campaigns=12, seed=17,
    ))
    return prepare_dataset(df, verbose=False)


def test_shuffled_labels_yield_no_signal(prepared, cfg):
    """With labels randomised, PR-AUC must collapse to the base rate."""
    rng = np.random.default_rng(0)
    scrambled = prepared.copy()
    scrambled[LABEL] = rng.permutation(scrambled[LABEL].to_numpy())

    split = temporal_split(scrambled)
    det = FakeReviewDetector(blocks=("text", "linguistic", "behaviour"),
                             cfg=cfg, oof_folds=2)
    det.fit(split.train)

    y = split.test[LABEL].to_numpy(int)
    ap = average_precision_score(y, det.predict_proba(split.test))
    base = y.mean()
    assert ap < base * 1.45, (
        f"PR-AUC {ap:.3f} on SHUFFLED labels vs base rate {base:.3f}: "
        "the pipeline is finding signal that cannot exist -- suspect leakage"
    )


def test_identifiers_are_not_model_features(prepared, cfg):
    """review_id / reviewer_id / product_id must never reach the model.

    Raw ids are the classic leak: a model that memorises known-bad reviewer
    ids scores well offline and catches nothing new in production.
    """
    split = temporal_split(prepared)
    det = FakeReviewDetector(blocks=("linguistic", "behaviour"), cfg=cfg)
    det.fit(split.train)
    forbidden = {"review_id", "reviewer_id", "product_id", "label"}
    assert not (forbidden & set(det.stage_b_columns_))


def test_label_column_absent_from_stage_b(prepared, cfg):
    split = temporal_split(prepared)
    det = FakeReviewDetector(blocks=("linguistic",), cfg=cfg)
    det.fit(split.train)
    assert LABEL not in det.stage_b_columns_


def test_constant_text_gives_behaviour_only_signal(prepared, cfg):
    """Blanking the text must not crash and must degrade text-only cleanly."""
    blanked = prepared.copy()
    blanked["text"] = "ok"
    blanked["title"] = ""
    split = temporal_split(blanked)
    det = FakeReviewDetector(blocks=("text",), cfg=cfg, oof_folds=2)
    det.fit(split.train)
    y = split.test[LABEL].to_numpy(int)
    ap = average_precision_score(y, det.predict_proba(split.test))
    assert ap < y.mean() * 1.4, "identical text should carry no signal"
