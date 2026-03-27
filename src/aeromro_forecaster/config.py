from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FORECAST_DIR = DATA_DIR / "forecasts"
DB_PATH = Path(os.getenv("AEROMRO_DB_PATH", DATA_DIR / "mro_forecast.db"))
DEFAULT_TOP_N = int(os.getenv("AEROMRO_TOP_N", "100"))
DEFAULT_LAST_DAYS = int(os.getenv("AEROMRO_LAST_DAYS", "730"))
DEFAULT_SALES_ROWS = int(os.getenv("AEROMRO_SALES_ROWS", "5000"))
DEFAULT_PRICE_ROWS = int(os.getenv("AEROMRO_PRICE_ROWS", "250000"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
