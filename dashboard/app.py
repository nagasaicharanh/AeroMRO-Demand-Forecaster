from __future__ import annotations

import sqlite3
import sys
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

try:
    import dash_bootstrap_components as dbc
except ImportError:  # pragma: no cover
    dbc = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aeromro_forecaster.config import DB_PATH, FORECAST_DIR
from aeromro_forecaster.llm_agent.agent import AgentUnavailable, ask


def read_skus() -> list[str]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id FROM demand GROUP BY id ORDER BY SUM(demand) DESC LIMIT 250").fetchall()
    return [row[0] for row in rows]


def read_history(sku_id: str) -> pd.DataFrame:
    if not DB_PATH.exists() or not sku_id:
        return pd.DataFrame(columns=["date", "demand"])
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT date, demand FROM demand WHERE id = ? ORDER BY date",
            conn,
            params=(sku_id,),
            parse_dates=["date"],
        )


def read_forecasts(sku_id: str) -> pd.DataFrame:
    frames = []
    for path in FORECAST_DIR.glob("*forecast.csv"):
        df = pd.read_csv(path)
        if "unique_id" in df.columns:
            frames.append(df[df["unique_id"] == sku_id])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def app_layout():
    skus = read_skus()
    default_sku = skus[0] if skus else None
    alert = None if skus else html.Div("No SQLite demand database found. Run the ETL after placing M5 files in data/raw.", className="alert alert-warning")
    return dbc.Container(
        [
            html.H1("AeroMRO Demand Forecaster", className="h3 mt-3 mb-3"),
            alert,
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Forecast Explorer",
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Dropdown(id="sku-select", options=[{"label": sku, "value": sku} for sku in skus], value=default_sku), md=6),
                                    dbc.Col(dcc.DatePickerRange(id="date-range"), md=6),
                                ],
                                className="my-3",
                            ),
                            dcc.Graph(id="forecast-chart"),
                        ],
                    ),
                    dcc.Tab(
                        label="Model Comparison",
                        children=[
                            dcc.Graph(id="metrics-chart"),
                            html.Div(id="quality-table", className="mt-3"),
                        ],
                    ),
                    dcc.Tab(
                        label="LLM Analyst Chat",
                        children=[
                            dbc.Input(id="chat-question", placeholder="Ask about demand, spikes, forecasts, or maintenance context.", className="my-3"),
                            dbc.Button("Ask", id="ask-button", n_clicks=0),
                            html.Pre(id="chat-answer", className="border rounded p-3 mt-3"),
                        ],
                    ),
                ]
            ),
        ],
        fluid=True,
    )


external = [dbc.themes.BOOTSTRAP] if dbc else []
app = Dash(__name__, external_stylesheets=external)
server = app.server
app.layout = app_layout


@app.callback(Output("forecast-chart", "figure"), Input("sku-select", "value"))
def update_forecast(sku_id: str):
    history = read_history(sku_id)
    forecasts = read_forecasts(sku_id)
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(go.Scatter(x=history["date"], y=history["demand"], mode="lines", name="Historical demand"))
    if not forecasts.empty:
        for model, df in forecasts.groupby("model" if "model" in forecasts.columns else "unique_id"):
            fig.add_trace(go.Scatter(x=df["ds"], y=df["yhat"], mode="lines", name=f"{model} forecast", line={"dash": "dash"}))
            if {"yhat_lower", "yhat_upper"}.issubset(df.columns):
                fig.add_trace(
                    go.Scatter(
                        x=list(df["ds"]) + list(df["ds"])[::-1],
                        y=list(df["yhat_upper"]) + list(df["yhat_lower"])[::-1],
                        fill="toself",
                        line={"color": "rgba(255,127,14,0)"},
                        name=f"{model} interval",
                    )
                )
    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Demand")
    return fig


@app.callback(Output("metrics-chart", "figure"), Input("sku-select", "value"))
def update_metrics(_):
    path = ROOT / "data" / "model_comparison.csv"
    if not path.exists():
        return go.Figure().update_layout(template="plotly_white", title="No model_comparison.csv found yet")
    df = pd.read_csv(path)
    long = df.melt(id_vars=["model"], value_vars=[c for c in ["MAE", "RMSE", "MASE"] if c in df.columns], var_name="metric", value_name="value")
    return px.bar(long, x="model", y="value", color="metric", barmode="group", template="plotly_white")


@app.callback(Output("quality-table", "children"), Input("sku-select", "value"))
def update_quality(_):
    if not DB_PATH.exists():
        return "No database available."
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT COUNT(*), COUNT(DISTINCT id), MIN(date), MAX(date) FROM demand").fetchone()
    return f"Rows: {rows[0]:,} | SKUs: {rows[1]:,} | Coverage: {rows[2]} to {rows[3]}"


@app.callback(Output("chat-answer", "children"), Input("ask-button", "n_clicks"), State("chat-question", "value"), prevent_initial_call=True)
def answer_question(_, question: str):
    if not question:
        return "Enter a question first."
    try:
        return ask(question)
    except AgentUnavailable as exc:
        return str(exc)
    except Exception as exc:
        return f"Agent error: {exc}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("AEROMRO_DASH_PORT", "8050")), debug=False)
