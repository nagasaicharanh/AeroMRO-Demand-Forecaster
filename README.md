# AeroMRO Demand Forecaster

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dashboard: Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-3F4F75?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Tracking: MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![LLM: Ollama](https://img.shields.io/badge/LLM-Ollama-111111)](https://ollama.com/)
[![Docker Compose](https://img.shields.io/badge/Container-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Last Commit](https://img.shields.io/github/last-commit/nagasaicharanh/AeroMRO-Demand-Forecaster?label=Last%20Commit)](https://github.com/nagasaicharanh/AeroMRO-Demand-Forecaster/commits/main)

Production-style local demand forecasting system for aircraft MRO inventory analytics, using the M5 Forecasting Competition dataset as a spare-parts demand proxy.

**Quick links:** [Architecture](#architecture) · [Setup](#setup) · [Run Locally](#run-locally) · [LLM Analyst](#llm-analyst) · [Docker](#docker) · [Screenshots](#screenshots)

## Architecture

```text
M5 CSVs -> pandas ETL -> SQLite demand table
                         |
                         +-> StatsForecast / XGBoost / LSTM -> MLflow + forecast CSVs
                         |
                         +-> LangGraph tools + ChromaDB RAG -> Ollama analyst assistant
                         |
                         +-> Plotly Dash dashboard
```

## Setup

```bash
conda create -n aero-forecast python=3.11
conda activate aero-forecast
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name aero-forecast
```

Download these Kaggle M5 files into `data/raw/`:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

## Run Locally

```bash
python -m aeromro_forecaster.etl.build_database --top-n 100 --last-days 730 --sales-rows 5000 --price-rows 250000
mlflow ui --port 5000
python -m aeromro_forecaster.models.train_baselines --top-n 25 --horizon 28 --n-jobs 1
python dashboard/app.py
```

Dashboard: <http://localhost:8050>  
MLflow: <http://localhost:5000>

## LLM Analyst

The dashboard chat uses a lightweight Ollama implementation by default and can run without LangGraph. Install Ollama, then pull the local chat model:

```bash
ollama pull llama3.2
```

Set `OLLAMA_BASE_URL` or `OLLAMA_CHAT_MODEL` if your Ollama server or model differs. If Ollama is not running, the chat tab falls back to a local tool summary from SQLite, forecast CSVs, and optional RAG context.

For LangGraph tool-calling mode, install the optional LangChain dependencies and set:

```bash
set AEROMRO_AGENT_BACKEND=langgraph
```

For optional RAG PDFs, also pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Optional RAG PDFs can be placed in `data/docs/`:

```bash
python -m aeromro_forecaster.llm_agent.build_rag
```

## Docker

```bash
docker compose up --build
```

Services:

- Dash: <http://localhost:8050>
- MLflow: <http://localhost:5000>
- Ollama: <http://localhost:11434>

Pull Ollama models into the compose volume:

```bash
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

## Screenshots

### Forecast Explorer

![Forecast Explorer](screenshots/Forecast%20Explorer.png)

### Model Comparison

![Model Comparison](screenshots/Model%20Comparison.png)

### LLM Analyst Chat

![LLM Analyst Chat](screenshots/LLM%20Analyst%20Chat.png)

## Tests

```bash
pytest
```

CI runs fixture-backed tests and a Docker build. Full M5 training is intentionally not required in CI.

## Repository Notes

Generated datasets, SQLite databases, MLflow runs, Chroma indexes, and model artifacts are ignored by git. Keep raw Kaggle data local under `data/raw/`.

For limited hardware, keep the ETL trimmed. `--sales-rows` and `--price-rows` limit CSV ingestion, `--top-n` limits the number of SKUs before reshaping, and `--last-days` limits the history window before the large melt/join step. The same defaults can be overridden with `AEROMRO_SALES_ROWS`, `AEROMRO_PRICE_ROWS`, `AEROMRO_TOP_N`, and `AEROMRO_LAST_DAYS`.
