"""Build a dataset: synthesise a marketplace, or normalise a real dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import CONFIG, PROCESSED_DIR, SynthConfig
from ..data.loaders import describe_coverage, load_generic, load_llm_fake_reviews
from ..data.synth import generate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "llm_fake_reviews", "generic"])
    ap.add_argument("--input", type=Path, help="input file for non-synthetic sources")
    ap.add_argument("--column-map", type=str,
                    help='JSON, e.g. \'{"review_body":"text","stars":"rating"}\'')
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR / "reviews.parquet")
    ap.add_argument("--n-reviews", type=int, default=CONFIG.synth.n_reviews)
    ap.add_argument("--fake-ratio", type=float, default=CONFIG.synth.fake_ratio)
    ap.add_argument("--campaigns", type=int, default=CONFIG.synth.n_campaigns)
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    args = ap.parse_args(argv)

    if args.source == "synthetic":
        cfg = SynthConfig(
            n_reviews=args.n_reviews,
            fake_ratio=args.fake_ratio,
            n_campaigns=args.campaigns,
            seed=args.seed,
        )
        df = generate(cfg)
        print(f"[dataset] synthesised {len(df):,} reviews "
              f"({df.label.mean():.1%} fake, {args.campaigns} campaigns)")
    elif args.source == "llm_fake_reviews":
        if not args.input:
            ap.error("--input is required for llm_fake_reviews")
        df = load_llm_fake_reviews(args.input)
    else:
        if not (args.input and args.column_map):
            ap.error("--input and --column-map are required for generic")
        df = load_generic(args.input, json.loads(args.column_map))

    cov = describe_coverage(df)
    print("[dataset] coverage:")
    for k, v in cov.items():
        print(f"    {k}: {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        try:
            df.to_parquet(args.out, index=False)
        except Exception as e:      # pyarrow absent -> CSV is fine here
            args.out = args.out.with_suffix(".csv")
            df.to_csv(args.out, index=False)
            print(f"[dataset] parquet unavailable ({type(e).__name__}); wrote CSV")
    else:
        df.to_csv(args.out, index=False)
    print(f"[dataset] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
