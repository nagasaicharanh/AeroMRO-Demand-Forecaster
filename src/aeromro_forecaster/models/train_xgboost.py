from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aeromro_forecaster.config import DB_PATH, FORECAST_DIR
from aeromro_forecaster.models.data import load_single_series


def train_xgboost_recursive(sku_id: str, db_path: Path, output_dir: Path, horizon: int) -> pd.DataFrame:
    try:
        import matplotlib.pyplot as plt
        import mlflow
        import shap
        from sklearn.preprocessing import StandardScaler
        from skforecast.ForecasterAutoreg import ForecasterAutoreg
        from skforecast.model_selection import backtesting_forecaster
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("Install skforecast, xgboost, shap, matplotlib, and mlflow for XGBoost training.") from exc

    series = load_single_series(sku_id, db_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecaster = ForecasterAutoreg(
        regressor=XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, objective="reg:squarederror"),
        lags=28,
        transformer_y=StandardScaler(),
    )
    with mlflow.start_run(run_name="XGBoost_recursive"):
        metrics, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=series,
            initial_train_size=int(len(series) * 0.8),
            steps=horizon,
            metric="mean_absolute_error",
            refit=False,
            verbose=False,
        )
        metric_value = float(metrics["mean_absolute_error"].iloc[0] if hasattr(metrics["mean_absolute_error"], "iloc") else metrics["mean_absolute_error"])
        mlflow.log_metric("mae", metric_value)
        mlflow.log_param("sku_id", sku_id)
        mlflow.log_param("n_estimators", 300)

        forecaster.fit(y=series)
        preds = forecaster.predict(steps=horizon)
        forecast = pd.DataFrame(
            {
                "model": "XGBoost",
                "unique_id": sku_id,
                "ds": pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"),
                "yhat": preds.to_numpy(),
            }
        )
        path = output_dir / "xgboost_forecast.csv"
        forecast.to_csv(path, index=False)
        mlflow.log_artifact(str(path))

        if hasattr(forecaster, "X_train_"):
            explainer = shap.TreeExplainer(forecaster.regressor)
            shap_values = explainer.shap_values(forecaster.X_train_)
            shap.summary_plot(shap_values, forecaster.X_train_, show=False)
            shap_path = output_dir / "xgboost_shap_importance.png"
            plt.tight_layout()
            plt.savefig(shap_path)
            plt.close()
            mlflow.log_artifact(str(shap_path))
        return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recursive XGBoost forecast for one SKU.")
    parser.add_argument("--sku-id", required=True)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=FORECAST_DIR)
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    forecast = train_xgboost_recursive(args.sku_id, args.db_path, args.output_dir, args.horizon)
    print(f"Wrote {len(forecast):,} forecast rows to {args.output_dir}")


if __name__ == "__main__":
    main()
