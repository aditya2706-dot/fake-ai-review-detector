"""End-to-end demo: train small, then show what the model actually caught.

    .venv/bin/python examples/demo.py

Prints the top-scored reviews with their reason codes, then the reviews the
model missed -- which is the more instructive half.
"""

from __future__ import annotations

import numpy as np

from frd.config import Config, SynthConfig
from frd.data.splits import temporal_split
from frd.data.synth import generate
from frd.evaluate import compute, threshold_table
from frd.explain import explain
from frd.models.pipeline import FakeReviewDetector
from frd.schema import LABEL
from frd.train import prepare_dataset

BAR = "=" * 78


def main() -> None:
    cfg = Config()
    cfg.text.word_max_features = 20_000
    cfg.text.char_max_features = 10_000
    cfg.model.lgbm_params["n_estimators"] = 300

    print(BAR); print("1. SYNTHESISE A MARKETPLACE"); print(BAR)
    df = generate(SynthConfig(n_reviews=12_000, n_products=300,
                              n_genuine_reviewers=2_000, n_campaigns=30, seed=42))
    print(f"{len(df):,} reviews | {df[LABEL].mean():.1%} inauthentic | "
          f"{df.reviewer_id.nunique():,} accounts | {df.product_id.nunique()} products")

    print(f"\n{BAR}"); print("2. FEATURES (causal: each row sees only its own past)"); print(BAR)
    prepared = prepare_dataset(df, verbose=True)
    split = temporal_split(prepared)

    print(f"\n{BAR}"); print("3. TRAIN + CALIBRATE + SET THRESHOLD"); print(BAR)
    det = FakeReviewDetector(blocks=("text", "linguistic", "behaviour"), cfg=cfg)
    det.fit(split.train)
    det.calibrate(split.val)

    y_val = split.val[LABEL].to_numpy(int)
    from frd.evaluate import pick_threshold_for_precision
    thr, p, r = pick_threshold_for_precision(y_val, det.predict_proba(split.val), 0.90)
    det.set_threshold(thr)
    print(f"threshold {thr:.3f} chosen on validation for >=90% precision "
          f"(val precision {p:.3f}, recall {r:.3f})")

    print(f"\n{BAR}"); print("4. TEST-SET PERFORMANCE"); print(BAR)
    y = split.test[LABEL].to_numpy(int)
    s = det.predict_proba(split.test)
    m = compute(y, s, thr)
    print(f"PR-AUC     {m.pr_auc:.4f}   (random = {m.baseline_pr_auc:.4f}, "
          f"lift {m.lift_over_random:.1f}x)")
    print(f"ROC-AUC    {m.roc_auc:.4f}")
    print(f"P@2%       {m.p_at_2pct:.3f}     precision {m.precision:.3f}  "
          f"recall {m.recall:.3f}")

    print(f"\n{BAR}"); print("5. OPERATING POINTS (per 10,000 reviews/day)"); print(BAR)
    t = threshold_table(y, s, daily_volume=10_000)
    print(t[t.threshold.isin([0.1, 0.3, 0.5, 0.7, 0.9])].to_string(index=False))

    print(f"\n{BAR}"); print("6. WHAT IT CAUGHT -- top 3 flagged, with reasons"); print(BAR)
    top_idx = np.argsort(-s)[:3]
    for e, row in zip(explain(det, split.test.iloc[top_idx]),
                      split.test.iloc[top_idx].itertuples()):
        print(f"\n  score {e.score:.3f} -> {e.decision.upper():18s} "
              f"(truth: {'FAKE' if row.label else 'GENUINE'})")
        print(f"  {row.rating:.0f} stars | \"{row.text[:88]}...\"")
        for reason in e.reasons[:4]:
            print(f"    * {reason}")

    print(f"\n{BAR}"); print("7. WHAT IT MISSED -- the instructive half"); print(BAR)
    missed = np.where((y == 1) & (s < thr))[0]
    if len(missed):
        worst = missed[np.argsort(s[missed])[:3]]
        print(f"  {len(missed)} of {int(y.sum())} fakes scored below threshold "
              f"({len(missed) / y.sum():.1%} missed)\n")
        for i in worst:
            row = split.test.iloc[i]
            print(f"  score {s[i]:.3f} | {row.rating:.0f} stars | "
                  f"author had {row.rv_prior_count:.0f} prior reviews")
            print(f"    \"{row.text[:88]}...\"")
        print("\n  These are the lone-wolf and high-effort fakes: specific text,")
        print("  no burst, no template reuse. Behavioural features cannot see them.")
        print("  Graph features were built for exactly this and do not catch them")
        print("  either -- a lone wolf has no co-review structure by definition.")
        print("  See ROADMAP phase 7 for what measuring that actually produced.")
    else:
        print("  nothing missed at this threshold")

    print(f"\n{BAR}"); print("8. TOP FEATURES"); print(BAR)
    print(det.feature_importance(top=10).to_string(index=False))


if __name__ == "__main__":
    main()
