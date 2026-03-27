from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from aeromro_forecaster.config import DB_PATH


def load_series_frame(db_path: Path = DB_PATH, top_n: int | None = 100) -> pd.DataFrame:
    query = "SELECT id AS unique_id, date AS ds, demand AS y FROM demand"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, parse_dates=["ds"])
    if top_n:
        top = df.groupby("unique_id")["y"].sum().nlargest(top_n).index
        df = df[df["unique_id"].isin(top)]
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def load_single_series(sku_id: str, db_path: Path = DB_PATH) -> pd.Series:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT date, demand FROM demand WHERE id = ? ORDER BY date",
            conn,
            params=(sku_id,),
            parse_dates=["date"],
        )
    if df.empty:
        raise ValueError(f"No demand history found for SKU {sku_id}")
    return pd.Series(df["demand"].to_numpy(), index=df["date"], name=sku_id)
