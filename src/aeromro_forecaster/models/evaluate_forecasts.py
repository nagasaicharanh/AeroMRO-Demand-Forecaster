from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from aeromro_forecaster.config import DB_PATH, FORECAST_DIR, PROJECT_ROOT
from aeromro_forecaster.models.metrics import mae, mase, rmse


def load_actuals(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT id AS unique_id, date AS ds, demand AS y FROM demand",
            conn,
            parse_dates=["ds"],
        )


def load_forecasts(forecast_dir: Path) -> pd.DataFrame:
    frames = []
    for path in forecast_dir.glob("*forecast.csv"):
        df = pd.read_csv(path, parse_dates=["ds"])
        if {"model", "unique_id", "ds", "yhat"}.issubset(df.columns):
            frames.append(df[["model", "unique_id", "ds", "yhat"]])
    if not frames:
        raise FileNotFoundError(f"No normalized forecast CSVs found in {forecast_dir}")
    return pd.concat(frames, ignore_index=True)


def evaluate(db_path: Path = DB_PATH, forecast_dir: Path = FORECAST_DIR, output_path: Path | None = None) -> pd.DataFrame:
    actuals = load_actuals(db_path)
    forecasts = load_forecasts(forecast_dir)
    joined = forecasts.merge(actuals, on=["unique_id", "ds"], how="inner")
    if joined.empty:
        raise ValueError("Forecast dates do not overlap with actual demand dates; cannot evaluate.")

    rows = []
    for model, group in joined.groupby("model"):
        train = actuals[actuals["ds"] < group["ds"].min()]["y"]
        rows.append(
            {
                "model": model,
                "MAE": mae(group["y"], group["yhat"]),
                "RMSE": rmse(group["y"], group["yhat"]),
                "MASE": mase(group["y"], group["yhat"], train, seasonality=7) if len(train) > 7 else float("nan"),
            }
        )
    result = pd.DataFrame(rows).sort_values("MAE")
    output_path = output_path or PROJECT_ROOT / "data" / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecast CSVs against actuals in SQLite.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--forecast-dir", type=Path, default=FORECAST_DIR)
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "data" / "model_comparison.csv")
    args = parser.parse_args()
    result = evaluate(args.db_path, args.forecast_dir, args.output_path)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
