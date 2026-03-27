FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY tests ./tests
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY . .
EXPOSE 8050 5000
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & python dashboard/app.py"]
