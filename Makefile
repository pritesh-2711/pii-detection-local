.PHONY: install prepare train test api docker clean help

help:
	@echo "PII Detection System - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install dependencies"
	@echo "  make prepare         Prepare training data"
	@echo ""
	@echo "Training:"
	@echo "  make train          Train the model"
	@echo "  make pipeline       Run complete pipeline (prepare + train)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run test suite"
	@echo ""
	@echo "Deployment:"
	@echo "  make api            Start API server"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean generated files"
	@echo "  make clean-models   Remove trained models"

install:
	pip install -r requirements.txt
	pip install -r requirements-api.txt

prepare:
	python src/data_preparation.py

train:
	python src/train.py

pipeline:
	python run_pipeline.py

test:
	python test_detector.py

api:
	python src/api.py --model-path ./models/best_model

api-prod:
	gunicorn -w 4 -b 0.0.0.0:5000 src.api:app

docker-build:
	docker build -t pii-detector:latest .

docker-run:
	docker run -p 5000:5000 pii-detector:latest

docker-run-gpu:
	docker run --gpus all -p 5000:5000 pii-detector:latest

clean:
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc src/*.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-models:
	rm -rf models/*

clean-data:
	rm -rf data/*.jsonl data/*.json

clean-all: clean clean-models clean-data
	@echo "All generated files removed"
