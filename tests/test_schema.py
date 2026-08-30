import pandas as pd
import pytest

from frd.schema import LABEL, REQUIRED_COLUMNS, TIMESTAMP, normalise


def _minimal(n=3):
    return pd.DataFrame({
        "review_id": [f"R{i}" for i in range(n)],
        "product_id": ["P1"] * n,
        "reviewer_id": ["U1"] * n,
        "rating": [5.0, 1.0, 3.0][:n],
        "text": ["good", "bad", "ok"][:n],
        "timestamp": pd.date_range("2025-01-03", periods=n, freq="-1D", tz="UTC"),
    })


def test_normalise_fills_optional_columns():
    out = normalise(_minimal())
    for col in ("title", "verified_purchase", "helpful_votes", "category"):
        assert col in out.columns


def test_normalise_sorts_by_time():
    out = normalise(_minimal())
    assert out[TIMESTAMP].is_monotonic_increasing


@pytest.mark.parametrize("missing", list(REQUIRED_COLUMNS))
def test_missing_required_column_raises(missing):
    df = _minimal().drop(columns=[missing])
    with pytest.raises(ValueError, match="missing required column"):
        normalise(df)


def test_unparseable_timestamp_raises():
    df = _minimal()
    df["timestamp"] = ["not a date", "also not", "nope"]
    with pytest.raises(ValueError, match="unparseable"):
        normalise(df)


def test_rating_is_clipped():
    df = _minimal()
    df["rating"] = [9.0, -3.0, 3.0]
    out = normalise(df)
    assert out["rating"].max() <= 5 and out["rating"].min() >= 1


def test_labels_preserved_as_int():
    df = _minimal().assign(label=[1, 0, 1])
    out = normalise(df)
    assert out[LABEL].dtype.kind == "i"
