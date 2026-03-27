from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from aeromro_forecaster.config import DB_PATH, FORECAST_DIR


def query_demand(sku_id: str, days: int = 30, db_path: Path = DB_PATH) -> str:
    """Fetch recent historical demand for a given SKU."""
    if not db_path.exists():
        return f"Demand database not found at {db_path}."
    safe_days = max(1, min(int(days), 365))
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT date, demand, rolling_mean_7, event_name_1
            FROM demand
            WHERE id = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            conn,
            params=(sku_id, safe_days),
        )
    if df.empty:
        return f"No demand history found for SKU {sku_id}."
    return df.sort_values("date").to_string(index=False)


def get_forecast(sku_id: str, model: str | None = None, forecast_dir: Path = FORECAST_DIR) -> str:
    """Return available 28-day ahead forecasts for a SKU."""
    if not forecast_dir.exists():
        return f"Forecast directory not found at {forecast_dir}."
    frames = []
    for path in forecast_dir.glob("*forecast.csv"):
        df = pd.read_csv(path)
        if "unique_id" not in df.columns:
            continue
        subset = df[df["unique_id"] == sku_id]
        if model and "model" in subset.columns:
            subset = subset[subset["model"].str.lower() == model.lower()]
        if not subset.empty:
            frames.append(subset)
    if not frames:
        return f"No forecast available for SKU {sku_id}."
    result = pd.concat(frames, ignore_index=True)
    cols = [col for col in ["model", "unique_id", "ds", "yhat", "yhat_lower", "yhat_upper"] if col in result.columns]
    return result[cols].to_string(index=False)


def rag_search(query: str, persist_directory: str = "data/chroma_db") -> str:
    """Search local MRO documents for context."""
    try:
        import chromadb
    except ImportError:
        return "ChromaDB is not installed."

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_collection("mro_manuals")
        results = collection.query(query_texts=[query], n_results=3)
    except Exception as exc:
        return f"No RAG collection available: {exc}"
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else "No relevant documents found."
