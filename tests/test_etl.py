from __future__ import annotations

import sqlite3

import pandas as pd

from aeromro_forecaster.etl.build_database import build_demand_frame, require_columns, write_sqlite


def fixture_frames():
    sales = pd.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_evaluation", "FOODS_1_002_CA_1_evaluation"],
            "item_id": ["FOODS_1_001", "FOODS_1_002"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            **{f"d_{i}": [i, i + 1] for i in range(1, 31)},
        }
    )
    calendar = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, 31)],
            "date": pd.date_range("2011-01-01", periods=30),
            "wm_yr_wk": [11101] * 30,
            "wday": [(i % 7) + 1 for i in range(30)],
            "month": [1] * 30,
            "year": [2011] * 30,
            "event_name_1": [None] * 29 + ["Promo"],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_002"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [1.25, 2.5],
        }
    )
    return sales, calendar, prices


def test_require_columns_reports_missing():
    df = pd.DataFrame({"id": [1]})
    try:
        require_columns(df, {"id", "date"}, "sample")
    except ValueError as exc:
        assert "date" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation failure")


def test_build_demand_frame_reshapes_and_features():
    sales, calendar, prices = fixture_frames()
    df = build_demand_frame(sales, calendar, prices)
    assert len(df) == 60
    assert {"lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", "is_weekend", "is_event"}.issubset(df.columns)
    first_sku = df[df["id"] == "FOODS_1_001_CA_1_evaluation"].reset_index(drop=True)
    assert pd.isna(first_sku.loc[0, "lag_7"])
    assert first_sku.loc[7, "lag_7"] == 1
    assert first_sku.loc[29, "is_event"] == 1


def test_build_demand_frame_can_trim_before_reshape():
    sales, calendar, prices = fixture_frames()
    df = build_demand_frame(sales, calendar, prices, top_n=1, last_days=7)

    assert len(df) == 7
    assert df["id"].nunique() == 1
    assert df["id"].iloc[0] == "FOODS_1_002_CA_1_evaluation"
    assert df["date"].min() == pd.Timestamp("2011-01-24")
    assert df["date"].max() == pd.Timestamp("2011-01-30")


def test_write_sqlite_creates_table_and_indexes(tmp_path):
    sales, calendar, prices = fixture_frames()
    df = build_demand_frame(sales, calendar, prices)
    db_path = tmp_path / "mro_forecast.db"
    write_sqlite(df, db_path)
    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM demand").fetchone()[0]
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(demand)").fetchall()}
    assert row_count == 60
    assert "idx_demand_id_date" in indexes
