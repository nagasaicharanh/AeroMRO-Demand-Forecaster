from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aeromro_forecaster.config import DB_PATH, FORECAST_DIR, PROJECT_ROOT
from aeromro_forecaster.models.data import load_series_frame
from aeromro_forecaster.models.metrics import mae, rmse


def train_autoarima(db_path: Path, output_dir: Path, top_n: int, horizon: int, n_jobs: int) -> pd.DataFrame:
    try:
        import mlflow
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, SeasonalNaive
    except ImportError as exc:
        raise RuntimeError("Install statsforecast and mlflow to train baseline models.") from exc

    df = load_series_frame(db_path, top_n=top_n)
    output_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="AutoARIMA"):
        sf = StatsForecast(
            models=[AutoARIMA(season_length=7), SeasonalNaive(season_length=7)],
            freq="D",
            n_jobs=n_jobs,
        )
        cv = sf.cross_validation(df=df, h=horizon, step_size=horizon, n_windows=3)
        auto_mae = (cv["y"] - cv["AutoARIMA"]).abs().mean()
        seasonal_mae = (cv["y"] - cv["SeasonalNaive"]).abs().mean()
        mlflow.log_metric("autoarima_mae", float(auto_mae))
        mlflow.log_metric("seasonal_naive_mae", float(seasonal_mae))
        sf.fit(df)
        forecast = sf.predict(h=horizon).reset_index()
        out = forecast.rename(columns={"ds": "ds"})
        out["model"] = "AutoARIMA"
        out = out.rename(columns={"AutoARIMA": "yhat"})
        cols = ["model", "unique_id", "ds", "yhat"] + [c for c in out.columns if c not in {"model", "unique_id", "ds", "yhat", "SeasonalNaive"}]
        out = out[cols]
        path = output_dir / "autoarima_forecast.csv"
        out.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
        return out


def train_seasonal_naive(db_path: Path, output_dir: Path, top_n: int, horizon: int, season_length: int = 7) -> pd.DataFrame:
    df = load_series_frame(db_path, top_n=top_n)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = []
    metrics = []
    for sku_id, group in df.groupby("unique_id", sort=False):
        group = group.sort_values("ds").reset_index(drop=True)
        if len(group) <= season_length:
            history = group["y"].to_numpy()
            holdout = pd.DataFrame()
        else:
            history = group["y"].iloc[:-season_length].to_numpy()
            holdout = group.iloc[-season_length:].copy()

        pattern = history[-season_length:] if len(history) >= season_length else history
        if len(pattern) == 0:
            pattern = np.array([0.0])
        yhat = np.resize(pattern, horizon).astype(float)
        forecasts.append(
            pd.DataFrame(
                {
                    "model": "SeasonalNaive",
                    "unique_id": sku_id,
                    "ds": pd.date_range(group["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"),
                    "yhat": yhat,
                }
            )
        )

        if not holdout.empty:
            holdout_pred = np.resize(pattern, len(holdout)).astype(float)
            metrics.append({"unique_id": sku_id, "MAE": mae(holdout["y"], holdout_pred), "RMSE": rmse(holdout["y"], holdout_pred)})

    forecast = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(columns=["model", "unique_id", "ds", "yhat"])
    forecast.to_csv(output_dir / "seasonal_naive_forecast.csv", index=False)
    if metrics:
        metric_df = pd.DataFrame(metrics)
        metric_df.to_csv(output_dir / "seasonal_naive_backtest_metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "SeasonalNaive",
                    "MAE": metric_df["MAE"].mean(),
                    "RMSE": metric_df["RMSE"].mean(),
                    "MASE": np.nan,
                }
            ]
        ).to_csv(PROJECT_ROOT / "data" / "model_comparison.csv", index=False)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StatsForecast baseline models.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=FORECAST_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--fallback", action="store_true", help="Use the built-in seasonal naive model instead of StatsForecast.")
    args = parser.parse_args()
    if args.fallback:
        forecast = train_seasonal_naive(args.db_path, args.output_dir, args.top_n, args.horizon)
    else:
        try:
            forecast = train_autoarima(args.db_path, args.output_dir, args.top_n, args.horizon, args.n_jobs)
        except RuntimeError as exc:
            print(f"{exc} Falling back to built-in seasonal naive model.")
            forecast = train_seasonal_naive(args.db_path, args.output_dir, args.top_n, args.horizon)
    print(f"Wrote {len(forecast):,} forecast rows to {args.output_dir}")


if __name__ == "__main__":
    main()
