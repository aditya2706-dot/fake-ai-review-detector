"""Train the detector and write the evaluation report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import CONFIG, PROCESSED_DIR, SynthConfig
from ..data.synth import generate
from ..schema import normalise
from ..train import ABLATIONS, run_all


def _load(path: Path | None, n_reviews: int, seed: int) -> pd.DataFrame:
    if path is None:
        print(f"[train] no --data given; synthesising {n_reviews:,} reviews")
        return generate(SynthConfig(n_reviews=n_reviews, seed=seed))
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return normalise(df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=None, help="dataset file; synthesises one if omitted")
    ap.add_argument("--n-reviews", type=int, default=CONFIG.synth.n_reviews)
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    ap.add_argument("--only", nargs="*", choices=list(ABLATIONS),
                    help="restrict to named ablations")
    ap.add_argument("--splits", nargs="*", default=["temporal", "cold_reviewer"],
                    choices=["temporal", "cold_reviewer"])
    ap.add_argument("--target-precision", type=float,
                    default=CONFIG.model.target_precision)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    default = PROCESSED_DIR / "reviews.parquet"
    data = args.data
    if data is None and default.exists():
        data = default
    elif data is None and default.with_suffix(".csv").exists():
        data = default.with_suffix(".csv")

    df = _load(data, args.n_reviews, args.seed)
    CONFIG.model.target_precision = args.target_precision

    ablations = (
        {k: v for k, v in ABLATIONS.items() if k in args.only}
        if args.only else ABLATIONS
    )
    report = run_all(
        df, cfg=CONFIG, ablations=ablations,
        split_kinds=tuple(args.splits), save_best=not args.no_save,
    )

    best = report.get("best_model")
    if best:
        print(f"\n[train] best: {best['name']} "
              f"(test PR-AUC {best['test_pr_auc']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
