"""Evaluation built around how the model is actually used.

Accuracy is nearly useless here. At an 18% fake rate, a model that flags
nothing scores 82%. What a trust-and-safety team needs to know is:

  * PR-AUC (average precision) -- ranking quality on the rare positive class.
    This is the headline number, not ROC-AUC, which flatters imbalanced tasks
    because the vast negative class makes the false-positive rate look small.
  * Precision@k -- if moderators can review 2% of daily volume, how many of
    that 2% are genuinely fake? This is the number that sets staffing.
  * Recall at a precision floor -- how much fraud is caught while keeping
    false accusations below what the business will tolerate. Wrongly deleting
    a real customer's review is a support ticket and a trust cost, so the
    threshold is a policy decision, not 0.5.
  * Calibration -- an "82% likely fake" score must mean 82%, or the automated
    tiers (auto-remove / queue / ignore) cannot be set rationally.
  * Slice metrics -- aggregate numbers hide the blind spots that matter, e.g.
    reviews from accounts with no prior history, where every behavioural
    feature is a sentinel value.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)


@dataclass
class Metrics:
    n: int
    positives: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    baseline_pr_auc: float          # = positive rate; PR-AUC of random ranking
    lift_over_random: float
    brier: float
    log_loss: float
    threshold: float
    precision: float
    recall: float
    f1: float
    flagged_rate: float
    tp: int
    fp: int
    fn: int
    tn: int
    p_at_1pct: float
    p_at_2pct: float
    p_at_5pct: float
    p_at_10pct: float
    r_at_5pct: float
    r_at_10pct: float

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def precision_at_k(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """Precision within the top `frac` of reviews by score (a review queue)."""
    k = max(1, int(round(len(s) * frac)))
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].mean())


def recall_at_k(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    k = max(1, int(round(len(s) * frac)))
    idx = np.argsort(-s, kind="stable")[:k]
    total = max(int(y.sum()), 1)
    return float(y[idx].sum() / total)


def pick_threshold_for_precision(
    y: np.ndarray, s: np.ndarray, target_precision: float = 0.90
) -> tuple[float, float, float]:
    """Lowest threshold whose precision still meets the target.

    Returns (threshold, precision, recall). Picking the LOWEST such threshold
    maximises recall subject to the precision floor -- catch as much fraud as
    possible without exceeding the tolerated false-accusation rate.
    """
    prec, rec, thr = precision_recall_curve(y, s)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    ok = np.where(prec[:-1] >= target_precision)[0]
    if len(ok) == 0:
        # Target unreachable: fall back to the best precision available.
        best = int(np.argmax(prec[:-1]))
        return float(thr[best]), float(prec[best]), float(rec[best])
    j = int(ok[np.argmax(rec[ok])])
    return float(thr[j]), float(prec[j]), float(rec[j])


def compute(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Metrics:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pos_rate = float(y.mean())
    pr_auc = float(average_precision_score(y, s))

    return Metrics(
        n=len(y),
        positives=int(y.sum()),
        positive_rate=pos_rate,
        roc_auc=float(roc_auc_score(y, s)),
        pr_auc=pr_auc,
        baseline_pr_auc=pos_rate,
        lift_over_random=pr_auc / pos_rate if pos_rate else float("nan"),
        brier=float(brier_score_loss(y, np.clip(s, 0, 1))),
        log_loss=float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))),
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        flagged_rate=float(pred.mean()),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        p_at_1pct=precision_at_k(y, s, 0.01),
        p_at_2pct=precision_at_k(y, s, 0.02),
        p_at_5pct=precision_at_k(y, s, 0.05),
        p_at_10pct=precision_at_k(y, s, 0.10),
        r_at_5pct=recall_at_k(y, s, 0.05),
        r_at_10pct=recall_at_k(y, s, 0.10),
    )


def threshold_table(
    y: np.ndarray, s: np.ndarray, daily_volume: int = 10_000
) -> pd.DataFrame:
    """Operating points, expressed as moderator workload.

    `daily_volume` scales the flagged counts into "reviews sent to a human per
    day", which is the form an ops lead can actually make a decision from.
    """
    y = np.asarray(y, int)
    rows = []
    for thr in np.arange(0.05, 1.0, 0.05):
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        flagged = tp + fp
        rows.append({
            "threshold": round(float(thr), 2),
            "flagged_rate": round(flagged / len(y), 4),
            "precision": round(tp / flagged, 4) if flagged else 0.0,
            "recall": round(tp / max(tp + fn, 1), 4),
            "flagged_per_day": int(round(flagged / len(y) * daily_volume)),
            "false_accusations_per_day": int(round(fp / len(y) * daily_volume)),
            "missed_fakes_per_day": int(round(fn / len(y) * daily_volume)),
        })
    return pd.DataFrame(rows)


def slice_metrics(
    df: pd.DataFrame, scores: np.ndarray, threshold: float,
    label_col: str = "label",
) -> pd.DataFrame:
    """Per-segment PR-AUC, to expose blind spots the headline number hides."""
    y = df[label_col].to_numpy(int)
    out = []

    def add(name: str, value: str, mask: np.ndarray) -> None:
        m = np.asarray(mask, bool)
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            return
        met = compute(y[m], scores[m], threshold)
        out.append({
            "slice": name, "value": value, "n": int(m.sum()),
            "pos_rate": round(float(y[m].mean()), 4),
            "pr_auc": round(met.pr_auc, 4),
            # Raw PR-AUC is not comparable across slices with different base
            # rates: 0.48 on a 3.5%-positive slice is 13x better than random,
            # while 0.48 on a 40%-positive slice is worse than random. Lift is
            # the only figure that can be read across rows of this table.
            "lift": round(met.lift_over_random, 2),
            "roc_auc": round(met.roc_auc, 4),
            "precision": round(met.precision, 4),
            "recall": round(met.recall, 4),
        })

    if "category" in df.columns:
        for c in df["category"].dropna().unique():
            add("category", str(c), df["category"] == c)

    if "rating" in df.columns:
        add("rating", "5", df["rating"] >= 4.5)
        add("rating", "4", (df["rating"] >= 3.5) & (df["rating"] < 4.5))
        add("rating", "2-3", (df["rating"] >= 1.5) & (df["rating"] < 3.5))
        add("rating", "1", df["rating"] < 1.5)

    if "verified_purchase" in df.columns:
        v = df["verified_purchase"].fillna(False).astype(bool)
        add("verified", "true", v)
        add("verified", "false", ~v)

    # The hardest and most important slice: accounts the model has no history
    # for, where every behavioural feature is a sentinel.
    if "rv_prior_count" in df.columns:
        pc = df["rv_prior_count"]
        add("reviewer_history", "0 (cold)", pc == 0)
        add("reviewer_history", "1-2", (pc >= 1) & (pc <= 2))
        add("reviewer_history", "3-9", (pc >= 3) & (pc <= 9))
        add("reviewer_history", "10+", pc >= 10)

    if "text" in df.columns:
        tl = df["text"].astype(str).str.len()
        add("text_length", "<80 chars", tl < 80)
        add("text_length", "80-200", (tl >= 80) & (tl < 200))
        add("text_length", "200+", tl >= 200)

    return pd.DataFrame(out)


def calibration_table(y: np.ndarray, s: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed fake rate per score decile."""
    y = np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": round(float(s[m].mean()), 4),
            "observed_fake_rate": round(float(y[m].mean()), 4),
            "gap": round(float(s[m].mean() - y[m].mean()), 4),
        })
    return pd.DataFrame(rows)


def save_figures(
    y: np.ndarray, s: np.ndarray, out_dir: str | Path, tag: str = "test"
) -> list[Path]:
    """PR curve, ROC, calibration, and score distribution."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, int)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    axes[0, 0].plot(rec, prec, lw=2, color="#c0392b")
    axes[0, 0].axhline(y.mean(), ls="--", c="grey",
                       label=f"random = {y.mean():.3f}")
    axes[0, 0].set(xlabel="Recall", ylabel="Precision",
                   title=f"Precision-Recall (AP = {ap:.3f})")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.3)

    fpr, tpr, _ = roc_curve(y, s)
    axes[0, 1].plot(fpr, tpr, lw=2, color="#2c3e50")
    axes[0, 1].plot([0, 1], [0, 1], ls="--", c="grey")
    axes[0, 1].set(xlabel="FPR", ylabel="TPR",
                   title=f"ROC (AUC = {roc_auc_score(y, s):.3f})")
    axes[0, 1].grid(alpha=.3)

    cal = calibration_table(y, s)
    axes[1, 0].plot(cal.mean_predicted, cal.observed_fake_rate, "o-",
                    color="#16a085")
    axes[1, 0].plot([0, 1], [0, 1], ls="--", c="grey")
    axes[1, 0].set(xlabel="Mean predicted", ylabel="Observed fake rate",
                   title="Calibration")
    axes[1, 0].grid(alpha=.3)

    axes[1, 1].hist(s[y == 0], bins=50, alpha=.6, label="genuine", density=True,
                    color="#2980b9")
    axes[1, 1].hist(s[y == 1], bins=50, alpha=.6, label="fake", density=True,
                    color="#c0392b")
    axes[1, 1].set(xlabel="Score", ylabel="Density",
                   title="Score distribution by class")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=.3)

    fig.suptitle(f"Fake review detector -- {tag}", fontsize=13)
    fig.tight_layout()
    p = out_dir / f"eval_{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths
