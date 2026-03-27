from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from aeromro_forecaster.config import DEFAULT_LAST_DAYS, DEFAULT_PRICE_ROWS, DEFAULT_SALES_ROWS, DEFAULT_TOP_N, DB_PATH, RAW_DIR


SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

SALES_REQUIRED = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
CALENDAR_REQUIRED = {"d", "date", "wday", "month", "year"}
PRICES_REQUIRED = {"store_id", "item_id", "wm_yr_wk", "sell_price"}


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _read_sales(path: Path, last_days: int | None, sales_rows: int | None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    day_cols = [col for col in header if col.startswith("d_")]
    if last_days is not None:
        if last_days < 1:
            raise ValueError("last_days must be greater than 0")
        day_cols = day_cols[-last_days:]
    id_cols = [col for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if col in header]
    return pd.read_csv(path, usecols=id_cols + day_cols, nrows=sales_rows)


def load_raw(
    raw_dir: Path,
    last_days: int | None = DEFAULT_LAST_DAYS,
    sales_rows: int | None = DEFAULT_SALES_ROWS,
    price_rows: int | None = DEFAULT_PRICE_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        SALES_FILE: raw_dir / SALES_FILE,
        CALENDAR_FILE: raw_dir / CALENDAR_FILE,
        PRICES_FILE: raw_dir / PRICES_FILE,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw M5 files in "
            f"{raw_dir}: {', '.join(missing)}. Download them from Kaggle first."
        )

    sales = _read_sales(paths[SALES_FILE], last_days=last_days, sales_rows=sales_rows)
    calendar = pd.read_csv(paths[CALENDAR_FILE])
    prices = pd.read_csv(paths[PRICES_FILE], nrows=price_rows)

    require_columns(sales, SALES_REQUIRED, SALES_FILE)
    require_columns(calendar, CALENDAR_REQUIRED, CALENDAR_FILE)
    require_columns(prices, PRICES_REQUIRED, PRICES_FILE)
    return sales, calendar, prices


def _trim_inputs(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int | None,
    last_days: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day_cols = [col for col in sales.columns if col.startswith("d_")]
    if not day_cols:
        raise ValueError("sales data contains no daily demand columns like d_1")

    selected_days = day_cols
    if last_days is not None:
        if last_days < 1:
            raise ValueError("last_days must be greater than 0")
        calendar_days = calendar[calendar["d"].isin(day_cols)].sort_values(
            "d", key=lambda col: col.str.replace(r"^d_", "", regex=True).astype(int)
        )
        selected_days = calendar_days["d"].tail(last_days).tolist()

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    if top_n is not None:
        if top_n < 1:
            raise ValueError("top_n must be greater than 0")
        totals = sales[selected_days].sum(axis=1).nlargest(top_n)
        sales = sales.loc[totals.index]

    sales = sales[id_cols + selected_days].copy()
    calendar = calendar[calendar["d"].isin(selected_days)].copy()

    keep_pairs = sales[["store_id", "item_id"]].drop_duplicates()
    keep_weeks = calendar[["wm_yr_wk"]].drop_duplicates() if "wm_yr_wk" in calendar.columns else pd.DataFrame()
    prices = prices.merge(keep_pairs, on=["store_id", "item_id"], how="inner")
    if not keep_weeks.empty and "wm_yr_wk" in prices.columns:
        prices = prices.merge(keep_weeks, on="wm_yr_wk", how="inner")

    return sales, calendar, prices


def build_demand_frame(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int | None = None,
    last_days: int | None = None,
) -> pd.DataFrame:
    sales, calendar, prices = _trim_inputs(sales, calendar, prices, top_n, last_days)
    day_cols = [col for col in sales.columns if col.startswith("d_")]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    demand = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="demand",
    )

    calendar_cols = [
        col
        for col in [
            "d",
            "date",
            "wm_yr_wk",
            "wday",
            "weekday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
        if col in calendar.columns
    ]
    demand = demand.merge(calendar[calendar_cols], on="d", how="left")
    demand = demand.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    demand["date"] = pd.to_datetime(demand["date"], errors="coerce")
    demand["demand"] = pd.to_numeric(demand["demand"], errors="coerce").fillna(0)
    demand = demand.sort_values(["id", "date"])

    grouped = demand.groupby("id", sort=False)["demand"]
    demand["lag_7"] = grouped.shift(7)
    demand["lag_28"] = grouped.shift(28)
    demand["rolling_mean_7"] = grouped.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    demand["rolling_mean_28"] = grouped.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    demand["rolling_std_7"] = grouped.shift(1).rolling(7, min_periods=2).std().reset_index(level=0, drop=True)
    demand["day_of_week"] = demand["date"].dt.dayofweek
    demand["is_weekend"] = demand["day_of_week"].isin([5, 6]).astype(int)
    demand["is_event"] = demand.get("event_name_1", pd.Series(index=demand.index)).notna().astype(int)
    demand["month"] = demand["date"].dt.month
    demand["year"] = demand["date"].dt.year
    return demand


def quality_report(df: pd.DataFrame) -> dict[str, object]:
    q1, q3 = df["demand"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_mask = (df["demand"] < q1 - 1.5 * iqr) | (df["demand"] > q3 + 1.5 * iqr)
    coverage = df.groupby("id")["date"].agg(["min", "max", "count"])
    return {
        "rows": int(len(df)),
        "sku_count": int(df["id"].nunique()),
        "null_counts": df.isnull().sum().astype(int).to_dict(),
        "outlier_rows": int(outlier_mask.sum()),
        "top_10_skus": df.groupby("id")["demand"].sum().nlargest(10).to_dict(),
        "min_date": str(coverage["min"].min().date()),
        "max_date": str(coverage["max"].max().date()),
    }


def write_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("demand", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_demand_id_date ON demand(id, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_demand_date ON demand(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_demand_store ON demand(store_id)")
        conn.commit()


def run(
    raw_dir: Path = RAW_DIR,
    db_path: Path = DB_PATH,
    top_n: int | None = DEFAULT_TOP_N,
    last_days: int | None = DEFAULT_LAST_DAYS,
    sales_rows: int | None = DEFAULT_SALES_ROWS,
    price_rows: int | None = DEFAULT_PRICE_ROWS,
) -> dict[str, object]:
    sales, calendar, prices = load_raw(raw_dir, last_days=last_days, sales_rows=sales_rows, price_rows=price_rows)
    demand = build_demand_frame(sales, calendar, prices, top_n=top_n, last_days=last_days)
    write_sqlite(demand, db_path)
    return quality_report(demand)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite demand database from M5 raw files.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Keep only the highest-demand N SKUs before reshaping.")
    parser.add_argument("--last-days", type=int, default=DEFAULT_LAST_DAYS, help="Keep only the most recent N daily columns before reshaping.")
    parser.add_argument("--sales-rows", type=int, default=DEFAULT_SALES_ROWS, help="Read only the first N sales rows from the wide M5 sales file.")
    parser.add_argument("--price-rows", type=int, default=DEFAULT_PRICE_ROWS, help="Read only the first N rows from sell_prices.csv.")
    args = parser.parse_args()
    report = run(
        args.raw_dir,
        args.db_path,
        top_n=args.top_n,
        last_days=args.last_days,
        sales_rows=args.sales_rows,
        price_rows=args.price_rows,
    )
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
