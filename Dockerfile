FROM python:3.10-slim

# Java for Spark model pyfunc
RUN apt-get update && apt-get install -y openjdk-11-jre && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving ./serving
COPY artifacts ./artifacts
COPY mlflow.db ./mlflow.db
COPY mlruns ./mlruns

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
