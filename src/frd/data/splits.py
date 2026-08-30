"""Dataset splitting.

Two regimes, because they answer different questions and a serious project
reports both:

1. TEMPORAL  -- train on the past, test on the future. This is the number that
   predicts production performance, because in production you always score
   reviews written after your training data. Fraud tactics drift, so this
   split is pessimistic and correct.

2. COLD-REVIEWER -- no reviewer appears in both train and test. This measures
   whether the model generalises to accounts it has never seen, or has merely
   memorised known-bad reviewer ids. A model that looks great temporally but
   collapses here is a reviewer-blocklist in disguise.

A plain random split is not offered. It overstates performance on this task by
a wide margin: the same campaign's reviews land on both sides of the split, so
the model recognises the template it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SplitConfig
from ..schema import REVIEWER_ID, TIMESTAMP


@dataclass(slots=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    kind: str

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("val", self.val),
                           ("test", self.test)):
            rows.append({
                "split": name,
                "rows": len(part),
                "positives": int(part["label"].sum()) if "label" in part else None,
                "pos_rate": round(float(part["label"].mean()), 4)
                if "label" in part else None,
                "start": part[TIMESTAMP].min(),
                "end": part[TIMESTAMP].max(),
            })
        return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> Split:
    """Chronological 70/15/15 by default. Assumes df is time-sorted."""
    cfg = cfg or SplitConfig()
    df = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    n = len(df)
    i_train = int(n * cfg.train_frac)
    i_val = int(n * (cfg.train_frac + cfg.val_frac))
    return Split(
        train=df.iloc[:i_train].copy(),
        val=df.iloc[i_train:i_val].copy(),
        test=df.iloc[i_val:].copy(),
        kind="temporal",
    )


def cold_reviewer_split(
    df: pd.DataFrame, cfg: SplitConfig | None = None, seed: int = 42
) -> Split:
    """Partition by reviewer so no account is shared across splits."""
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(seed)
    # np.asarray(..., dtype=object): pandas returns a StringArray, which
    # numpy cannot safely shuffle in place.
    reviewers = np.asarray(df[REVIEWER_ID].astype(str).unique(), dtype=object)
    rng.shuffle(reviewers)
    n = len(reviewers)
    i_train = int(n * cfg.train_frac)
    i_val = int(n * (cfg.train_frac + cfg.val_frac))

    train_ids = set(reviewers[:i_train])
    val_ids = set(reviewers[i_train:i_val])
    test_ids = set(reviewers[i_val:])

    rid = df[REVIEWER_ID].astype(str)
    return Split(
        train=df[rid.isin(train_ids)].copy(),
        val=df[rid.isin(val_ids)].copy(),
        test=df[rid.isin(test_ids)].copy(),
        kind="cold_reviewer",
    )
