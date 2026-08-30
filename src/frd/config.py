"""Central configuration. Every path and hyper-parameter lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
FIGURE_DIR = ARTIFACT_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


@dataclass(slots=True)
class SynthConfig:
    """Shape of the simulated marketplace."""

    n_products: int = 900
    n_genuine_reviewers: int = 6_000
    n_reviews: int = 40_000
    fake_ratio: float = 0.18          # realistic marketplace incidence
    n_campaigns: int = 70             # coordinated paid-review pushes
    days: int = 540
    seed: int = RANDOM_SEED


@dataclass(slots=True)
class SplitConfig:
    """Time-ordered split. Never random -- see docs/METHODOLOGY.md."""

    train_frac: float = 0.70
    val_frac: float = 0.15
    # remainder is test


@dataclass(slots=True)
class TextFeatureConfig:
    word_max_features: int = 60_000
    word_ngram: tuple[int, int] = (1, 2)
    char_max_features: int = 40_000
    char_ngram: tuple[int, int] = (3, 5)
    min_df: int = 3
    sublinear_tf: bool = True


@dataclass(slots=True)
class ModelConfig:
    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "n_estimators": 700,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 40,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
            "verbosity": -1,
        }
    )
    logreg_C: float = 4.0
    # Moderation queues have finite capacity: we tune the decision threshold
    # to hit this precision, rather than defaulting to 0.5.
    target_precision: float = 0.90


@dataclass(slots=True)
class Config:
    synth: SynthConfig = field(default_factory=SynthConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    text: TextFeatureConfig = field(default_factory=TextFeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = RANDOM_SEED


CONFIG = Config()
