"""Training orchestration: prepare -> split -> ablate -> tune -> report.

The ablation is not optional decoration. Behavioural feature engineering is
the expensive part of this system (it needs reviewer identity, timestamps, and
a chronological pass over history). The only way to justify that cost is to
show what the model scores without it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import evaluate as ev
from .config import CONFIG, Config
from .data.splits import Split, cold_reviewer_split, temporal_split
from .features.behavior import BEHAVIOUR_COLUMNS, build_behaviour_features
from .features.graph import GRAPH_COLUMNS, build_graph_features
from .features.text import LINGUISTIC_COLUMNS, LinguisticFeatures
from .models.pipeline import FakeReviewDetector
from .schema import LABEL

# The ablation ladder. Each rung adds one signal family so the marginal
# contribution of each is visible, rather than a single fused number.
ABLATIONS: dict[str, tuple[str, ...]] = {
    "text_only": ("text",),
    "linguistic_only": ("linguistic",),
    "behaviour_only": ("behaviour",),
    "graph_only": ("graph",),
    "text_plus_linguistic": ("text", "linguistic"),
    # The rung that pays for the graph block. `full_fusion` is deliberately
    # kept as the pre-graph fusion so the marginal contribution of the graph
    # features is a subtraction between two rows of the same table, not a
    # comparison against a number from an older run.
    "full_fusion": ("text", "linguistic", "behaviour"),
    "behaviour_plus_graph": ("behaviour", "graph"),
    "full_fusion_plus_graph": ("text", "linguistic", "behaviour", "graph"),
}


@dataclass
class ExperimentResult:
    name: str
    blocks: tuple[str, ...]
    split_kind: str
    backend: str
    threshold: float
    val: dict
    test: dict
    fit_seconds: float
    detector: FakeReviewDetector | None = field(default=None, repr=False)


def prepare_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Attach behavioural and linguistic feature columns to the raw frame.

    Behavioural features are computed over the WHOLE corpus before splitting.
    That is deliberate and not leakage: every one of them looks strictly
    backwards in time (see features/behavior.py), so a test-period review sees
    only train-period history -- precisely what it would see in production
    when the scorer queries the live reviews table.
    """
    if verbose:
        print(f"[prepare] {len(df):,} reviews")
    t0 = time.time()
    beh = build_behaviour_features(df, verbose=verbose)
    if verbose:
        print(f"[prepare] behavioural block: {beh.shape[1]} cols "
              f"in {time.time() - t0:.1f}s")

    t0 = time.time()
    gph = build_graph_features(df, verbose=verbose)
    if verbose:
        print(f"[prepare] graph block: {gph.shape[1]} cols "
              f"in {time.time() - t0:.1f}s")

    t0 = time.time()
    ling = pd.DataFrame(
        LinguisticFeatures().transform(df), columns=LINGUISTIC_COLUMNS,
        index=df.index,
    )
    if verbose:
        print(f"[prepare] linguistic block: {ling.shape[1]} cols "
              f"in {time.time() - t0:.1f}s")

    return pd.concat([df, beh, gph, ling], axis=1)


def make_split(df: pd.DataFrame, kind: str, cfg: Config = CONFIG) -> Split:
    if kind == "temporal":
        return temporal_split(df, cfg.split)
    if kind == "cold_reviewer":
        return cold_reviewer_split(df, cfg.split, seed=cfg.seed)
    raise ValueError(f"unknown split kind: {kind}")


def run_experiment(
    split: Split,
    blocks: tuple[str, ...],
    name: str,
    cfg: Config = CONFIG,
    verbose: bool = True,
) -> ExperimentResult:
    """Fit one ablation rung, tune its threshold on val, score on test."""
    t0 = time.time()
    det = FakeReviewDetector(blocks=blocks, cfg=cfg)
    det.fit(split.train)
    fit_s = time.time() - t0

    y_val = split.val[LABEL].to_numpy(int)

    # Validation does two jobs: it maps raw scores onto real probabilities, and
    # it sets the operating threshold. Both are model-selection concerns, so
    # both belong here and neither may touch test. Tuning a threshold on test
    # is the second-most-common way results on this task get inflated (the
    # first is leaky behavioural features).
    det.calibrate(split.val, y_val, kind="sigmoid")
    s_val = det.predict_proba(split.val)

    thr, p_at_thr, r_at_thr = ev.pick_threshold_for_precision(
        y_val, s_val, cfg.model.target_precision
    )
    det.set_threshold(thr)

    y_test = split.test[LABEL].to_numpy(int)
    s_test = det.predict_proba(split.test)

    val_m = ev.compute(y_val, s_val, thr).as_dict()
    test_m = ev.compute(y_test, s_test, thr).as_dict()
    # Mean absolute calibration gap on test, in probability points.
    cal = ev.calibration_table(y_test, s_test)
    test_m["calibration_mae"] = round(float(cal["gap"].abs().mean()), 4)

    if verbose:
        print(
            f"  {name:22s} [{det.backend_:14s}] "
            f"val PR-AUC {val_m['pr_auc']:.4f} | "
            f"test PR-AUC {test_m['pr_auc']:.4f} | "
            f"test P@2% {test_m['p_at_2pct']:.3f} | "
            f"thr {thr:.3f} -> P {test_m['precision']:.3f} "
            f"R {test_m['recall']:.3f} | "
            f"calMAE {test_m['calibration_mae']:.3f} | {fit_s:.1f}s"
        )

    return ExperimentResult(
        name=name, blocks=blocks, split_kind=split.kind,
        backend=det.backend_, threshold=thr, val=val_m, test=test_m,
        fit_seconds=round(fit_s, 2), detector=det,
    )


def run_all(
    df: pd.DataFrame,
    cfg: Config = CONFIG,
    ablations: dict[str, tuple[str, ...]] | None = None,
    split_kinds: tuple[str, ...] = ("temporal", "cold_reviewer"),
    save_best: bool = True,
    verbose: bool = True,
) -> dict:
    """Full study: both split regimes x every ablation rung.

    Returns a JSON-serialisable report and writes artefacts to disk.
    """
    ablations = ablations or ABLATIONS
    prepared = prepare_dataset(df, verbose=verbose)

    report: dict = {
        "dataset": {
            "n_reviews": int(len(prepared)),
            "positive_rate": round(float(prepared[LABEL].mean()), 4),
            "n_reviewers": int(prepared["reviewer_id"].nunique()),
            "n_products": int(prepared["product_id"].nunique()),
            "behaviour_features": len(BEHAVIOUR_COLUMNS),
            "graph_features": len(GRAPH_COLUMNS),
            "linguistic_features": len(LINGUISTIC_COLUMNS),
        },
        "config": {
            "target_precision": cfg.model.target_precision,
            "seed": cfg.seed,
        },
        "results": {},
    }

    best: tuple[float, ExperimentResult | None] = (-1.0, None)
    best_split: Split | None = None

    for kind in split_kinds:
        split = make_split(prepared, kind, cfg)
        if verbose:
            print(f"\n[split: {kind}]")
            print(split.summary().to_string(index=False))
            print()
        report["results"][kind] = {"split_summary": json.loads(
            split.summary().astype(str).to_json(orient="records")
        ), "ablations": {}}

        for name, blocks in ablations.items():
            res = run_experiment(split, blocks, name, cfg, verbose)
            report["results"][kind]["ablations"][name] = {
                "blocks": list(res.blocks),
                "backend": res.backend,
                "threshold": round(res.threshold, 4),
                "fit_seconds": res.fit_seconds,
                "val": res.val,
                "test": res.test,
            }
            # The temporal split is the production-realistic one, so the
            # shipped model is chosen on it -- and chosen on VALIDATION, not
            # test.
            #
            # This used to select on `res.test["pr_auc"]`, which is the same
            # mistake as tuning a threshold on test, one level up: picking
            # among 8 correlated architectures by their test score leaks test
            # into the shipped artifact and makes the headline number a
            # maximum over 8 draws rather than an estimate.
            #
            # It was not academic. `full_fusion` and `full_fusion_plus_graph`
            # sit within 0.004 PR-AUC of each other, and across two corpus
            # draws the graph block's contribution measured -0.0002, +0.0038,
            # +0.0018 and -0.0033 -- indistinguishable from zero, with the
            # sign flipping between val and test on *both* corpora. So the
            # shipped architecture was effectively being decided by a
            # test-set coin flip, in a repo whose own METHODOLOGY condemns
            # exactly that. Test is now only ever reported.
            if kind == "temporal" and res.val["pr_auc"] > best[0]:
                best = (res.val["pr_auc"], res)
                best_split = split

    # ---- artefacts for the winning model ----
    if best[1] is not None and best_split is not None:
        res = best[1]
        det = res.detector
        assert det is not None
        y_test = best_split.test[LABEL].to_numpy(int)
        s_test = det.predict_proba(best_split.test)

        report["best_model"] = {
            "name": res.name,
            "blocks": list(res.blocks),
            "split": "temporal",
            # Both are recorded so it is visible which one did the choosing.
            "selected_on": "val_pr_auc",
            "val_pr_auc": res.val["pr_auc"],
            "test_pr_auc": res.test["pr_auc"],
        }
        report["threshold_table"] = json.loads(
            ev.threshold_table(y_test, s_test).to_json(orient="records")
        )
        report["calibration"] = json.loads(
            ev.calibration_table(y_test, s_test).to_json(orient="records")
        )
        report["slices"] = json.loads(
            ev.slice_metrics(best_split.test, s_test, det.threshold_)
            .to_json(orient="records")
        )
        imp = det.feature_importance(top=30)
        report["feature_importance"] = json.loads(imp.to_json(orient="records"))

        ev.save_figures(y_test, s_test, cfg_figure_dir(), tag=res.name)

        if save_best:
            model_path = Path(cfg_model_dir()) / "detector.joblib"
            det.save(model_path)
            report["best_model"]["artifact"] = str(model_path)
            if verbose:
                print(f"\n[saved] {model_path}")

    report_path = Path(cfg_report_dir()) / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if verbose:
        print(f"[saved] {report_path}")

    return report


# Small indirections so tests can monkeypatch output locations.
def cfg_model_dir() -> Path:
    from .config import MODEL_DIR
    return MODEL_DIR


def cfg_report_dir() -> Path:
    from .config import REPORT_DIR
    return REPORT_DIR


def cfg_figure_dir() -> Path:
    from .config import FIGURE_DIR
    return FIGURE_DIR
