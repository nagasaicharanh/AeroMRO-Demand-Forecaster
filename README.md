<div align="center">

# ✈️ AeroMRO Demand Forecaster

### Local demand forecasting and analyst assistant for aircraft MRO inventory

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Plotly%20Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Experiment Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Forecasting](https://img.shields.io/badge/Forecasting-StatsForecast%20%2B%20XGBoost-4C8EDA?style=for-the-badge)](https://nixtlaverse.nixtla.io/statsforecast/)
[![LLM Assistant](https://img.shields.io/badge/LLM-Ollama-111111?style=for-the-badge)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)

Production-style forecasting workflow using the M5 Forecasting Competition dataset as a spare-parts demand proxy, with ETL, model training, experiment tracking, and an interactive dashboard assistant.
</div>

## 🎯 Key Features

- **End-to-End Pipeline**: M5 CSV ingestion, demand-table generation, forecasting, MLflow tracking, and dashboard visualization.
- **Multiple Model Families**: Baseline training support for statistical and ML approaches (StatsForecast, XGBoost, LSTM workflows).
- **Analyst Assistant**: Ollama-backed dashboard chat with optional LangGraph tool-calling and optional RAG context.
- **Practical Resource Controls**: Tunable ETL limits (`--sales-rows`, `--price-rows`, `--top-n`, `--last-days`) for constrained hardware.
- **Local-First Deployment**: Runs fully on local machine; optional Docker setup for consistent environments.

## 🖼️ Application Interface

### Forecast Explorer
![Forecast Explorer](screenshots/Forecast%20Explorer.png)

### Model Comparison
![Model Comparison](screenshots/Model%20Comparison.png)

### LLM Analyst Chat
![LLM Analyst Chat](screenshots/LLM%20Analyst%20Chat.png)

## 🏗️ Architecture

```text
M5 CSVs -> pandas ETL -> SQLite demand table
                         |
                         +-> StatsForecast / XGBoost / LSTM -> MLflow + forecast CSVs
                         |
                         +-> LangGraph tools + ChromaDB RAG -> Ollama analyst assistant
                         |
                         +-> Plotly Dash dashboard
```

## 📦 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Ingestion / ETL** | pandas + SQLite | Convert large M5 CSVs into queryable demand data |
| **Forecasting** | StatsForecast, XGBoost, LSTM | Compare demand forecasting approaches |
| **Experiment Tracking** | MLflow | Track runs, metrics, and artifacts |
| **Dashboard** | Plotly Dash | Interactive monitoring and analysis UI |
| **Assistant Runtime** | Ollama + optional LangGraph | Local chat assistant with tool-calling option |
| **Optional RAG** | ChromaDB + local docs (`data/docs/`) | Ground answers with local reference material |
| **Containerization** | Docker Compose | Reproducible local deployment |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Kaggle M5 files in `data/raw/`:
  - `sales_train_evaluation.csv`
  - `calendar.csv`
  - `sell_prices.csv`

### Installation

```bash
conda create -n aero-forecast python=3.11
conda activate aero-forecast
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name aero-forecast
```

### Run Locally

```bash
python -m aeromro_forecaster.etl.build_database --top-n 100 --last-days 730 --sales-rows 5000 --price-rows 250000
mlflow ui --port 5000
python -m aeromro_forecaster.models.train_baselines --top-n 25 --horizon 28 --n-jobs 1
python dashboard/app.py
```

- Dashboard: <http://localhost:8050>
- MLflow: <http://localhost:5000>

## 🤖 LLM Analyst

Pull the default local chat model:

```bash
ollama pull llama3.2
```

Set `OLLAMA_BASE_URL` or `OLLAMA_CHAT_MODEL` if needed. If Ollama is unavailable, the chat tab falls back to a local tool summary from SQLite, forecast CSVs, and optional RAG context.

For LangGraph tool-calling mode:

```bash
set AEROMRO_AGENT_BACKEND=langgraph
```

For optional RAG PDFs, also pull embeddings and build index:

```bash
ollama pull nomic-embed-text
python -m aeromro_forecaster.llm_agent.build_rag
```

## 🐳 Docker

```bash
docker compose up --build
```

Services:

- Dash: <http://localhost:8050>
- MLflow: <http://localhost:5000>
- Ollama: <http://localhost:11434>

Pull Ollama models into compose volume:

```bash
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

## ✅ Tests

```bash
pytest
```

CI runs fixture-backed tests and Docker build checks. Full M5 training is intentionally not required in CI.

## 📝 Repository Notes

Generated datasets, SQLite databases, MLflow runs, Chroma indexes, and model artifacts are ignored by git. Keep raw Kaggle files local under `data/raw/`.

For limited hardware, ETL controls can be overridden with environment variables: `AEROMRO_SALES_ROWS`, `AEROMRO_PRICE_ROWS`, `AEROMRO_TOP_N`, and `AEROMRO_LAST_DAYS`.
