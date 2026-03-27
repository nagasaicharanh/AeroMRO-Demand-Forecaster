from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aeromro_forecaster.config import DB_PATH, FORECAST_DIR, PROJECT_ROOT
from aeromro_forecaster.models.data import load_single_series


def make_windows(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(values)):
        x.append(values[i - seq_len : i])
        y.append(values[i])
    return np.asarray(x, dtype=np.float32)[..., None], np.asarray(y, dtype=np.float32)[..., None]


def train_lstm(sku_id: str, db_path: Path, output_dir: Path, epochs: int, horizon: int, seq_len: int = 56) -> pd.DataFrame:
    try:
        import mlflow
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install torch and mlflow for LSTM training.") from exc

    class StackedLSTM(nn.Module):
        def __init__(self, input_size: int = 1, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    series = load_single_series(sku_id, db_path)
    values = series.to_numpy(dtype=np.float32)
    mean = values.mean()
    std = values.std() or 1.0
    scaled = (values - mean) / std
    x, y = make_windows(scaled, seq_len)
    if len(x) < 10:
        raise ValueError("Not enough history to train LSTM")

    split = int(len(x) * 0.8)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[:split]), torch.from_numpy(y[:split])), batch_size=32, shuffle=True)
    val_x = torch.from_numpy(x[split:])
    val_y = torch.from_numpy(y[split:])
    model = StackedLSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_fn = nn.HuberLoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="StackedLSTM"):
        mlflow.log_param("sku_id", sku_id)
        mlflow.log_param("seq_len", seq_len)
        for epoch in range(epochs):
            model.train()
            last_loss = 0.0
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_x), val_y).item()) if len(val_x) else last_loss
            scheduler.step(val_loss)
            mlflow.log_metric("train_loss", last_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        history = list(scaled[-seq_len:])
        preds = []
        model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                xb = torch.tensor(np.asarray(history[-seq_len:], dtype=np.float32)[None, :, None])
                pred = float(model(xb).item())
                history.append(pred)
                preds.append(pred * std + mean)

        weights_path = PROJECT_ROOT / "models" / "lstm_weights.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        mlflow.log_artifact(str(weights_path))

    forecast = pd.DataFrame(
        {
            "model": "LSTM",
            "unique_id": sku_id,
            "ds": pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"),
            "yhat": preds,
        }
    )
    forecast.to_csv(output_dir / "lstm_forecast.csv", index=False)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM forecast for one SKU.")
    parser.add_argument("--sku-id", required=True)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=FORECAST_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    forecast = train_lstm(args.sku_id, args.db_path, args.output_dir, args.epochs, args.horizon)
    print(f"Wrote {len(forecast):,} forecast rows to {args.output_dir}")


if __name__ == "__main__":
    main()
