FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY run_data_pipeline.py .
COPY run_training_pipeline.py .
COPY run_benchmarking.py .
COPY test_detector.py .
COPY example_client.py .
COPY models/ ./models/

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/best_model

CMD ["python", "src/api.py", "--model-path", "/app/models/best_model", "--host", "0.0.0.0", "--port", "5000"]