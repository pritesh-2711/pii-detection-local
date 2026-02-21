.PHONY: help install data train pipeline test api api-prod \
        benchmark download-deps \
        docker-build docker-run docker-run-gpu \
        clean clean-models clean-data clean-all

help:
	@echo "PII Detection System"
	@echo ""
	@echo "Setup:"
	@echo "  make install            Install all dependencies"
	@echo "  make download-deps      Download spaCy models for benchmarking"
	@echo ""
	@echo "Data:"
	@echo "  make data               Download datasets + prepare train/val/test splits"
	@echo "  make data SKIP=1        Skip download, re-run consolidation + preparation only"
	@echo ""
	@echo "Training:"
	@echo "  make train              Download base model + fine-tune"
	@echo "  make train SKIP=1       Skip model download, run fine-tuning only"
	@echo "  make pipeline           Full pipeline: data + train"
	@echo ""
	@echo "Evaluation:"
	@echo "  make test               Run test suite against models/best_model"
	@echo "  make benchmark          Benchmark our model vs spaCy vs Presidio"
	@echo "  make benchmark N=500    Benchmark on first 500 test records"
	@echo ""
	@echo "Deployment:"
	@echo "  make api                Start API server (dev mode)"
	@echo "  make api-prod           Start API server via gunicorn (4 workers)"
	@echo "  make docker-build       Build Docker image"
	@echo "  make docker-run         Run Docker container"
	@echo "  make docker-run-gpu     Run Docker container with GPU passthrough"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              Remove __pycache__ and .pyc files"
	@echo "  make clean-models       Remove trained model checkpoints"
	@echo "  make clean-data         Remove data splits"
	@echo "  make clean-all          Remove everything generated"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	pip install -r requirements.txt

download-deps:
	python -m spacy download en_core_web_trf
	python -m spacy download en_core_web_lg

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

data:
ifdef SKIP
	python run_data_pipeline.py --skip-download
else
	python run_data_pipeline.py
endif

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

train:
ifdef SKIP
	python run_training_pipeline.py --skip-download
else
	python run_training_pipeline.py
endif

pipeline: data train

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

test:
	python test_detector.py

benchmark:
ifdef N
	python run_benchmarking.py --max-records $(N)
else
	python run_benchmarking.py
endif

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

api:
	python src/api.py --model-path ./models/best_model

api-prod:
	gunicorn -w 4 -b 0.0.0.0:5000 "src.api:create_app('./models/best_model')"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:
	docker build -t pii-detector:latest .

docker-run:
	docker run -p 5000:5000 pii-detector:latest

docker-run-gpu:
	docker run --gpus all -p 5000:5000 pii-detector:latest

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-models:
	rm -rf models/checkpoints models/best_model

clean-data:
	rm -rf data/ pii_datasets/

clean-all: clean clean-models clean-data
	rm -rf benchmark_results/
	@echo "All generated files removed."