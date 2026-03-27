from __future__ import annotations

import sqlite3

import pandas as pd

from aeromro_forecaster.llm_agent.tools import get_forecast, query_demand


def test_query_demand_uses_sqlite(tmp_path):
    db_path = tmp_path / "mro.db"
    with sqlite3.connect(db_path) as conn:
        pd.DataFrame(
            {
                "id": ["SKU_1", "SKU_1"],
                "date": ["2024-01-01", "2024-01-02"],
                "demand": [3, 4],
                "rolling_mean_7": [None, 3],
                "event_name_1": [None, "Holiday"],
            }
        ).to_sql("demand", conn, index=False)
    result = query_demand("SKU_1", db_path=db_path)
    assert "2024-01-02" in result
    assert "Holiday" in result


def test_get_forecast_handles_missing_sku(tmp_path):
    pd.DataFrame({"model": ["AutoARIMA"], "unique_id": ["SKU_1"], "ds": ["2024-01-03"], "yhat": [5]}).to_csv(
        tmp_path / "autoarima_forecast.csv", index=False
    )
    assert "No forecast" in get_forecast("SKU_2", forecast_dir=tmp_path)
    assert "AutoARIMA" in get_forecast("SKU_1", forecast_dir=tmp_path)
