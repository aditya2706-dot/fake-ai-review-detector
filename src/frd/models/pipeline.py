"""The detector: a two-stage stacked model.

WHY STACKED RATHER THAN ONE BIG MODEL
-------------------------------------
The two signal families have incompatible shapes. Text is ~100k sparse TF-IDF
dimensions where a linear model is excellent and a tree is slow and weak.
Behaviour is ~75 dense, heavily non-linear, interaction-rich columns where a
gradient-boosted tree is excellent and a linear model is hopeless. Concatenating
them and picking one learner means crippling one family.

So:

  Stage A (text)      TF-IDF word + char n-grams -> calibrated LogisticRegression
                      -> a single scalar `text_score`
  Stage B (fusion)    GBDT over [behavioural | linguistic | text_score]

Stage A's contribution to Stage B's training set is computed OUT OF FOLD,
grouped by product, so Stage B never sees a text score that Stage A produced
while having memorised that exact review. Grouping by product (rather than
plain k-fold) keeps a single coordinated campaign inside one fold -- otherwise
Stage A recognises the campaign's template from the other folds and hands Stage
B an unrealistically good score.

ABLATION IS A FIRST-CLASS FEATURE
---------------------------------
`blocks=` selects which families are active, so text-only / behaviour-only /
fused variants are the same code path. Comparing them is the only way to know
whether the expensive behavioural machinery earns its place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from ..config import CONFIG, Config
from ..features.behavior import BEHAVIOUR_COLUMNS, build_behaviour_features
from ..features.graph import GRAPH_COLUMNS, build_graph_features
from ..features.text import (
    LINGUISTIC_COLUMNS, LinguisticFeatures, TextColumnSelector,
    make_char_vectorizer, make_word_vectorizer,
)
from ..schema import LABEL, PRODUCT_ID, TIMESTAMP

Block = Literal["text", "linguistic", "behaviour", "graph"]
ALL_BLOCKS: tuple[Block, ...] = ("text", "linguistic", "behaviour", "graph")


class ColumnSubset(BaseEstimator, TransformerMixin):
    """Select named columns as a dense float32 array, tolerating absences."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnSubset":
        self.available_ = [c for c in self.columns if c in X.columns]
        if not self.available_:
            raise ValueError(
                f"none of the requested columns are present: {self.columns[:5]}..."
            )
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.available_, dtype=object)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.reindex(columns=self.available_).to_numpy(dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_text_pipeline(cfg: Config = CONFIG) -> Pipeline:
    """TF-IDF word + char union -> logistic regression."""
    union = FeatureUnion([
        ("word", Pipeline([
            ("sel", TextColumnSelector(join_title=True)),
            ("tfidf", make_word_vectorizer(cfg.text)),
        ])),
        ("char", Pipeline([
            ("sel", TextColumnSelector(join_title=True)),
            ("tfidf", make_char_vectorizer(cfg.text)),
        ])),
    ])
    return Pipeline([
        ("features", union),
        ("clf", LogisticRegression(
            C=cfg.model.logreg_C,
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
        )),
    ])


def _make_booster(cfg: Config):
    """LightGBM when its native lib loads, sklearn's HistGB otherwise.

    LightGBM needs OpenMP (`libomp`), which is absent on a stock macOS. Rather
    than make the project undeployable on the most common dev laptop, fall
    back to sklearn's histogram booster -- same algorithm family, no system
    dependency, within a point or two of LightGBM on tabular data this size.
    """
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        return LGBMClassifier(**cfg.model.lgbm_params), "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

        p = cfg.model.lgbm_params
        return (
            HistGradientBoostingClassifier(
                max_iter=p["n_estimators"],
                learning_rate=p["learning_rate"],
                max_leaf_nodes=p["num_leaves"],
                min_samples_leaf=p["min_child_samples"],
                l2_regularization=p["reg_lambda"],
                random_state=p["random_state"],
                early_stopping=False,
            ),
            "sklearn_histgb",
        )


@dataclass
class FitReport:
    backend: str
    blocks: tuple[str, ...]
    n_train: int
    n_features_stage_b: int
    oof_folds: int
    text_auc_oof: float | None = None
    extras: dict = field(default_factory=dict)


class FakeReviewDetector:
    """Stacked text + behaviour detector with a tuned decision threshold."""

    def __init__(
        self,
        blocks: Iterable[Block] = ALL_BLOCKS,
        cfg: Config = CONFIG,
        oof_folds: int = 4,
    ):
        self.blocks = tuple(blocks)
        unknown = set(self.blocks) - set(ALL_BLOCKS)
        if unknown:
            raise ValueError(f"unknown block(s): {unknown}")
        if not self.blocks:
            raise ValueError("at least one feature block is required")
        self.cfg = cfg
        self.oof_folds = oof_folds

        self.text_model_: Pipeline | None = None
        self.stage_b_ = None
        self.stage_b_columns_: list[str] = []
        self.backend_: str = ""
        self.threshold_: float = 0.5
        self.calibrator_ = None
        self.calibration_kind_: str = "none"
        self.report_: FitReport | None = None

    # ---------------- internals ----------------

    def _tabular_columns(self, X: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        if "linguistic" in self.blocks:
            cols += [c for c in LINGUISTIC_COLUMNS if c in X.columns]
        if "behaviour" in self.blocks:
            cols += [c for c in BEHAVIOUR_COLUMNS if c in X.columns]
        if "graph" in self.blocks:
            cols += [c for c in GRAPH_COLUMNS if c in X.columns]
        return cols

    def _ensure_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Materialise any active feature block the caller did not precompute.

        Training goes through `frd.train.prepare_dataset`, which attaches both
        blocks up front. Serving hands over a bare review. Rather than make
        callers remember which columns to build, the model builds whatever is
        missing -- using the same functions training used, so the two paths
        cannot drift apart.
        """
        need_ling = (
            "linguistic" in self.blocks
            and not any(c in X.columns for c in LINGUISTIC_COLUMNS)
        )
        need_beh = (
            "behaviour" in self.blocks
            and not any(c in X.columns for c in BEHAVIOUR_COLUMNS)
        )
        need_graph = (
            "graph" in self.blocks
            and not any(c in X.columns for c in GRAPH_COLUMNS)
        )
        if not (need_ling or need_beh or need_graph):
            return X

        additions: list[pd.DataFrame] = []
        if need_ling:
            additions.append(pd.DataFrame(
                LinguisticFeatures().transform(X),
                columns=LINGUISTIC_COLUMNS, index=X.index,
            ))
        if need_beh or need_graph:
            # Both blocks need chronological order; restore the caller's row
            # order afterwards so scores line up with input rows.
            ts = pd.to_datetime(X[TIMESTAMP], utc=True)
            ordered = ts.is_monotonic_increasing
            order = None if ordered else ts.sort_values(kind="mergesort").index
            source = X if ordered else X.loc[order]
            for needed, build in (
                (need_beh, build_behaviour_features),
                (need_graph, build_graph_features),
            ):
                if not needed:
                    continue
                block = build(source)
                additions.append(block if ordered else block.reindex(X.index))
        return pd.concat([X, *additions], axis=1)

    def _stage_b_matrix(
        self, X: pd.DataFrame, text_score: np.ndarray | None
    ) -> tuple[np.ndarray, list[str]]:
        X = self._ensure_columns(X)

        if self.stage_b_columns_:
            # Fitted: reproduce the exact columns and order seen during fit.
            # Selecting by name from a recorded list is what makes a silent
            # column-order change impossible rather than merely unlikely.
            tab_cols = [c for c in self.stage_b_columns_ if c != "text_score"]
        else:
            tab_cols = self._tabular_columns(X)

        parts: list[np.ndarray] = []
        names: list[str] = []

        if tab_cols:
            missing = [c for c in tab_cols if c not in X.columns]
            if missing:
                raise RuntimeError(
                    f"cannot rebuild {len(missing)} stage-B column(s), "
                    f"e.g. {missing[:5]}"
                )
            arr = X.reindex(columns=tab_cols).to_numpy(dtype=np.float32)
            parts.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
            names += tab_cols

        if text_score is not None:
            parts.append(text_score.reshape(-1, 1).astype(np.float32))
            names.append("text_score")

        if not parts:
            raise RuntimeError("no stage-B features produced")
        return np.hstack(parts), names

    def _oof_text_score(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Out-of-fold text probabilities, grouped by product."""
        groups = X[PRODUCT_ID].astype(str).to_numpy()
        n_groups = len(np.unique(groups))
        folds = max(2, min(self.oof_folds, n_groups))
        gkf = GroupKFold(n_splits=folds)
        oof = np.zeros(len(X), dtype=np.float32)

        for tr, te in gkf.split(X, y, groups=groups):
            if len(np.unique(y[tr])) < 2:
                oof[te] = float(y[tr].mean()) if len(tr) else 0.5
                continue
            m = build_text_pipeline(self.cfg)
            m.fit(X.iloc[tr], y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]

        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        auc = (
            float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
        )
        return oof, auc

    # ---------------- public API ----------------

    def fit(self, train: pd.DataFrame, y: np.ndarray | None = None) -> "FakeReviewDetector":
        y = train[LABEL].to_numpy(dtype=int) if y is None else np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training data contains a single class")

        text_oof: np.ndarray | None = None
        text_auc: float | None = None
        if "text" in self.blocks:
            text_oof, text_auc = self._oof_text_score(train, y)
            # Refit stage A on all of train for inference.
            self.text_model_ = build_text_pipeline(self.cfg)
            self.text_model_.fit(train, y)

        Xb, names = self._stage_b_matrix(train, text_oof)
        self.stage_b_columns_ = names

        booster, backend = _make_booster(self.cfg)
        self.backend_ = backend
        # Class imbalance: reweight positives rather than resample, so the
        # calibration of the scores stays interpretable.
        pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))
        sw = np.where(y == 1, pos_w, 1.0)
        booster.fit(Xb, y, sample_weight=sw)
        self.stage_b_ = booster

        self.report_ = FitReport(
            backend=backend,
            blocks=self.blocks,
            n_train=len(train),
            n_features_stage_b=Xb.shape[1],
            oof_folds=self.oof_folds,
            text_auc_oof=text_auc,
        )
        return self

    def decision_score(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated stage-B probability -- correct for RANKING only."""
        if self.stage_b_ is None:
            raise RuntimeError("model is not fitted")
        text_score = (
            self.text_model_.predict_proba(X)[:, 1].astype(np.float32)
            if self.text_model_ is not None else None
        )
        Xb, _ = self._stage_b_matrix(X, text_score)
        # Guard against a column-order drift between fit and score.
        if Xb.shape[1] != len(self.stage_b_columns_):
            raise RuntimeError(
                f"stage-B width mismatch: got {Xb.shape[1]}, "
                f"expected {len(self.stage_b_columns_)}"
            )
        return self.stage_b_.predict_proba(Xb)[:, 1]

    def calibrate(
        self, val: pd.DataFrame, y: np.ndarray | None = None,
        kind: str = "sigmoid",
    ) -> "FakeReviewDetector":
        """Map raw scores onto true probabilities using held-out data.

        Necessary because stage B is trained with positive-class sample
        weights: reweighting is what makes the model attend to the rare class,
        but it also inflates its raw output, so a raw 0.8 does not mean an 80%
        chance of being fake. Anything that consumes the number as a
        probability -- tiered auto-removal, expected-cost decisions, showing a
        confidence to a moderator -- needs this step.

        `sigmoid` (Platt) is the default because a validation split has few
        positives and isotonic regression overfits them; `isotonic` is
        available when the calibration set is large.

        Ranking metrics (PR-AUC, ROC-AUC, precision@k) are unchanged by any
        monotone calibration -- only the probability values and hence the
        meaning of a fixed threshold move.
        """
        y = val[LABEL].to_numpy(int) if y is None else np.asarray(y, int)
        raw = self.decision_score(val)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration data contains a single class")

        if kind == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(raw, y)
        elif kind == "sigmoid":
            cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            cal.fit(_logit(raw).reshape(-1, 1), y)
        else:
            raise ValueError(f"unknown calibration kind: {kind}")

        self.calibrator_ = cal
        self.calibration_kind_ = kind
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.decision_score(X)
        if self.calibrator_ is None:
            return raw
        if self.calibration_kind_ == "isotonic":
            return np.clip(self.calibrator_.predict(raw), 0.0, 1.0)
        return self.calibrator_.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(X) >= thr).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = float(threshold)

    def feature_importance(self, top: int = 25) -> pd.DataFrame:
        if self.stage_b_ is None:
            raise RuntimeError("model is not fitted")
        if hasattr(self.stage_b_, "feature_importances_"):
            imp = np.asarray(self.stage_b_.feature_importances_, dtype=float)
        else:  # HistGB exposes no native importances
            return pd.DataFrame(
                {"feature": self.stage_b_columns_, "importance": np.nan}
            )
        s = pd.DataFrame({"feature": self.stage_b_columns_, "importance": imp})
        s["share"] = s.importance / max(s.importance.sum(), 1e-9)
        return s.sort_values("importance", ascending=False).head(top).reset_index(drop=True)

    # ---------------- persistence ----------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        meta = {
            "blocks": list(self.blocks),
            "backend": self.backend_,
            "threshold": self.threshold_,
            "calibration": self.calibration_kind_,
            "stage_b_features": len(self.stage_b_columns_),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "FakeReviewDetector":
        return joblib.load(Path(path))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Platt scaling operates in log-odds space, so map probabilities there."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))
