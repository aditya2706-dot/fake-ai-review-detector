import numpy as np
import pandas as pd

from frd.features.text import LINGUISTIC_COLUMNS, LinguisticFeatures
from frd.schema import normalise


def _frame(texts, ratings=None):
    n = len(texts)
    return normalise(pd.DataFrame({
        "review_id": [f"R{i}" for i in range(n)],
        "product_id": ["P1"] * n, "reviewer_id": ["U1"] * n,
        "rating": ratings or [5.0] * n,
        "text": texts,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
    }))


def test_shape_and_finiteness():
    out = LinguisticFeatures().transform(_frame(["good", "", "a" * 500]))
    assert out.shape == (3, len(LINGUISTIC_COLUMNS))
    assert np.isfinite(out).all()


def test_empty_text_does_not_divide_by_zero():
    out = LinguisticFeatures().transform(_frame([""]))
    assert np.isfinite(out).all()


def test_specificity_separates_concrete_from_generic():
    """The core linguistic hypothesis: real reviews cite concrete detail."""
    idx = LINGUISTIC_COLUMNS.index("lx_specificity_score")
    generic = "Amazing product! Highly recommend! Best purchase ever!"
    concrete = ("Measured 31cm not the 29cm listed, and it ran 6 hours "
                "rather than the advertised 9 hours.")
    out = LinguisticFeatures().transform(_frame([generic, concrete]))
    assert out[1, idx] > out[0, idx]


def test_superlative_density_separates_promo_text():
    idx = LINGUISTIC_COLUMNS.index("lx_superlative_density")
    out = LinguisticFeatures().transform(_frame([
        "The best, most amazing, perfect, excellent, outstanding item.",
        "It arrived on Tuesday and the box was slightly dented.",
    ]))
    assert out[0, idx] > out[1, idx]


def test_promo_phrase_detection():
    idx = LINGUISTIC_COLUMNS.index("lx_promo_phrase_count")
    out = LinguisticFeatures().transform(_frame([
        "highly recommend, worth every penny, must buy",
        "It is fine.",
    ]))
    assert out[0, idx] >= 3
    assert out[1, idx] == 0


def test_transformer_is_picklable():
    """Serving loads the fitted pipeline from disk, so no lambdas allowed."""
    import pickle
    from frd.features.text import TextColumnSelector
    for obj in (LinguisticFeatures(), TextColumnSelector()):
        assert pickle.loads(pickle.dumps(obj)) is not None
