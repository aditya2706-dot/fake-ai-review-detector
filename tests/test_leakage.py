"""Guards against the failure modes that silently inflate results.

These are negative-control tests: they check the pipeline CANNOT find signal
where none exists. A pipeline that scores well on shuffled labels has a bug
(usually an identifier leaking in, or a target-derived feature), and without a
test like this the bug reads as a great result.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from frd.config import Config, SynthConfig
from frd.data.splits import temporal_split
from frd.data.synth import (
    FAKE_CLOSERS_POSITIVE,
    POSITIVE_OPENERS,
    generate,
)
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


# --- oracles inside the generator itself -----------------------------------
#
# The tests above check the model cannot find signal that is not there. This
# one checks the *data* does not contain signal a real marketplace would never
# hand you. That is the same class of failure -- a result inflated by
# something other than the thing being measured -- and it is easier to
# introduce, because writing generator vocabulary does not feel like writing a
# feature.


def _sentiment_oracle_rate(df):
    """Share of reviews carrying five-star copy despite a 1-2 star rating."""
    pattern = "|".join(
        re.escape(phrase)
        for phrase in FAKE_CLOSERS_POSITIVE + POSITIVE_OPENERS
    )
    low = df[df["rating"] <= 2]
    hit = low["text"].astype(str).str.contains(pattern, regex=True)
    return (
        float(hit[low[LABEL] == 1].mean()),
        float(hit[low[LABEL] == 0].mean()),
    )


def test_generator_has_no_sentiment_rating_oracle():
    """Low-rated fakes must not be identifiable by five-star copy alone.

    This held a real defect. Slogan openers and the paid-review closer were
    drawn from all-positive pools regardless of rating, and fakes used them far
    more often than genuine reviews did. The result: 68.3% of 1-2 star fakes
    carried five-star copy against 0.4% of genuine ones, and the hand-written
    rule `rating <= 2 AND a glowing closer` reached 97.9% precision over 12.6%
    of all fakes with no model involved at all.

    An attack campaign that writes "Highly recommend to everyone!" under a
    one-star rating is not a thing real farms do, so every point the model
    earned from that contradiction was borrowed against nothing. Sentiment now
    follows rating in both classes; the separation is slogan *density*, which
    is a structural property the generator is entitled to simulate.
    """
    df = generate(SynthConfig(n_reviews=8_000, n_campaigns=25, seed=5))
    fake_rate, genuine_rate = _sentiment_oracle_rate(df)

    assert fake_rate < 0.02, (
        f"{fake_rate:.1%} of low-rated fakes carry five-star copy; the "
        "sentiment/rating contradiction is back"
    )
    # Symmetry is the real invariant: whatever rate the two classes have, one
    # must not be orders of magnitude above the other, or the contradiction is
    # a label oracle no matter which direction it points.
    assert abs(fake_rate - genuine_rate) < 0.05


def test_sloganeering_still_separates_the_classes():
    """The fix must not have removed the signal along with the oracle.

    Paid copy really is more slogan-dense than a real review, in both
    sentiments. That is what the generator is meant to teach; if this drops to
    parity, the fix went too far and the text features have nothing to learn.
    """
    df = generate(SynthConfig(n_reviews=8_000, n_campaigns=25, seed=5))
    pattern = "|".join(re.escape(p) for p in FAKE_CLOSERS_POSITIVE + POSITIVE_OPENERS)
    high = df[df["rating"] >= 4]
    hit = high["text"].astype(str).str.contains(pattern, regex=True)
    fake, genuine = hit[high[LABEL] == 1].mean(), hit[high[LABEL] == 0].mean()
    assert fake > genuine + 0.15, (
        f"slogan density no longer separates classes ({fake:.1%} vs {genuine:.1%})"
    )


def test_shipped_model_is_selected_on_validation_not_test(cfg, tmp_path):
    """The architecture choice must not be made on the test set.

    Tuning a threshold on test is the mistake this project's METHODOLOGY calls
    out; choosing *which of eight architectures to ship* on test is the same
    mistake one level up, and it makes the reported headline a maximum over
    eight correlated draws rather than an estimate.

    It mattered here. `full_fusion` and `full_fusion_plus_graph` sit within
    0.004 PR-AUC, and across two corpus draws the graph block measured
    -0.0002, +0.0038, +0.0018 and -0.0033 -- with val and test disagreeing
    about the winner on both corpora. The shipped architecture was decided by
    a test-set coin flip.
    """
    from frd.train import run_all

    df = generate(SynthConfig(
        n_reviews=1500, n_products=40, n_genuine_reviewers=250,
        n_campaigns=8, seed=17,
    ))
    report = run_all(
        df, cfg,
        # Two rungs that differ, so there is a choice to get wrong.
        ablations={"behaviour_only": ("behaviour",),
                   "full_fusion": ("text", "linguistic", "behaviour")},
        split_kinds=("temporal",),
        save_best=False,
        verbose=False,
    )

    best = report["best_model"]
    assert best["selected_on"] == "val_pr_auc"

    abl = report["results"]["temporal"]["ablations"]
    val_winner = max(abl, key=lambda k: abl[k]["val"]["pr_auc"])
    assert best["name"] == val_winner, (
        f"shipped {best['name']} but validation preferred {val_winner}"
    )
    # And the report must still carry the test number, unused for choosing.
    assert "test_pr_auc" in best
