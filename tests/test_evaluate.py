import numpy as np

from frd import evaluate as ev


def test_precision_at_k_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, .1, .1, .1, .1, .1, .1, .1, .1])
    assert ev.precision_at_k(y, s, 0.2) == 1.0


def test_recall_at_k():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    s = np.array([0.9, 0.85, .1, .1, .1, .1, .1, .1, .05, .05])
    assert ev.recall_at_k(y, s, 0.2) == 0.5


def test_metrics_on_perfect_separation():
    y = np.array([0] * 90 + [1] * 10)
    s = np.array([0.01] * 90 + [0.99] * 10)
    m = ev.compute(y, s, threshold=0.5)
    assert m.pr_auc > 0.99 and m.roc_auc > 0.99
    assert m.precision == 1.0 and m.recall == 1.0
    assert m.fp == 0 and m.fn == 0


def test_random_scores_give_pr_auc_near_base_rate():
    """PR-AUC of a random ranking is the positive rate. Sanity anchor."""
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.15).astype(int)
    s = rng.random(20_000)
    m = ev.compute(y, s, 0.5)
    assert abs(m.pr_auc - 0.15) < 0.02
    assert abs(m.roc_auc - 0.5) < 0.03


def test_threshold_targets_requested_precision():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.2).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0.2, 0.18, 4000), 0, 1)
    thr, prec, rec = ev.pick_threshold_for_precision(y, s, 0.9)
    assert prec >= 0.9 - 1e-9
    assert 0.0 < thr < 1.0
    assert rec > 0


def test_unreachable_precision_falls_back_without_crashing():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.1).astype(int)
    s = rng.random(2000)          # no signal -> 0.99 precision impossible
    thr, prec, rec = ev.pick_threshold_for_precision(y, s, 0.99)
    assert 0.0 <= prec <= 1.0 and np.isfinite(thr)


def test_threshold_table_columns():
    rng = np.random.default_rng(3)
    y = (rng.random(1000) < 0.2).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 1000), 0, 1)
    t = ev.threshold_table(y, s, daily_volume=10_000)
    assert {"threshold", "precision", "recall", "flagged_per_day",
            "false_accusations_per_day"} <= set(t.columns)
    assert (t.recall.diff().dropna() <= 1e-9).all()   # recall falls as thr rises
