"""Score reviews from a file or a single inline review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import MODEL_DIR
from ..explain import explain
from ..models.pipeline import FakeReviewDetector
from ..schema import normalise
from ..train import prepare_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=MODEL_DIR / "detector.joblib")
    ap.add_argument("--data", type=Path, help="CSV/parquet of reviews to score")
    ap.add_argument("--text", type=str, help="score one review body inline")
    ap.add_argument("--rating", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=10, help="rows to print")
    ap.add_argument("--out", type=Path, help="write scored rows here")
    args = ap.parse_args(argv)

    if not args.model.exists():
        print(f"error: no model at {args.model}; run `make train` first")
        return 1
    det = FakeReviewDetector.load(args.model)

    if args.text:
        df = normalise(pd.DataFrame([{
            "review_id": "inline-1", "product_id": "P0", "reviewer_id": "U0",
            "rating": args.rating, "text": args.text,
            "timestamp": pd.Timestamp.now(tz="UTC"),
        }]))
    elif args.data:
        df = normalise(
            pd.read_parquet(args.data) if args.data.suffix == ".parquet"
            else pd.read_csv(args.data)
        )
    else:
        ap.error("provide --text or --data")

    prepared = prepare_dataset(df, verbose=False)
    exps = explain(det, prepared)
    scores = [e.score for e in exps]
    prepared = prepared.assign(
        fake_score=scores,
        decision=[e.decision for e in exps],
    )

    if args.text:
        print(json.dumps(exps[0].as_dict(), indent=2))
    else:
        cols = ["review_id", "rating", "fake_score", "decision"]
        top = prepared.sort_values("fake_score", ascending=False).head(args.top)
        print(top[cols].to_string(index=False))
        print(f"\n[score] {len(prepared):,} reviews | "
              f"flagged {(prepared.decision != 'allow').mean():.2%} "
              f"at threshold {det.threshold_:.3f}")

    if args.out:
        keep = [c for c in ("review_id", "product_id", "reviewer_id", "rating",
                            "text", "fake_score", "decision") if c in prepared]
        prepared[keep].to_csv(args.out, index=False)
        print(f"[score] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
