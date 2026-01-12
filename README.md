# PII Detection System

Production-ready PII detection system using fine-tuned transformer models.

## Project Structure

```
pii-detector/
├── data/                      # Training datasets
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── label_mapping.json
├── models/                    # Trained models
│   └── best_model/
├── src/
│   ├── data_preparation.py   # Data collection and preprocessing
│   ├── train.py              # Model training
│   ├── inference.py          # Prediction/inference
│   └── api.py                # REST API server
├── run_pipeline.py           # End-to-end pipeline
├── test_detector.py          # Test suite
├── requirements.txt          # Python dependencies
├── requirements-api.txt      # API dependencies
└── Dockerfile                # Docker deployment
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

This will:
- Download and prepare datasets (CoNLL-2003 + synthetic PII)
- Fine-tune BERT model for PII detection
- Evaluate on test set
- Save model to `./models/best_model`

### 3. Test the Model

```bash
python test_detector.py
```

### 4. Use in Python Code

```python
from src.inference import PIIDetector

detector = PIIDetector(model_path="./models/best_model")

text = "My email is john@example.com and phone is 555-1234"
result = detector.detect(text)

if result:
    print(f"PII detected: {result['pii_types']}")
    print(f"Entities: {result['entities']}")
else:
    print("No PII detected")
```

## Detailed Usage

### Data Preparation

Collect and prepare training data:

```bash
python src/data_preparation.py
```

This creates:
- `data/train.jsonl` - Training set
- `data/val.jsonl` - Validation set
- `data/test.jsonl` - Test set
- `data/label_mapping.json` - Label definitions

### Model Training

Train a custom PII detection model:

```bash
python src/train.py
```

Options:
- `model_name`: Base model (bert-base-cased, roberta-base, microsoft/deberta-v3-base)
- `num_epochs`: Training epochs (default: 5)
- `batch_size`: Batch size (default: 16)
- `learning_rate`: Learning rate (default: 5e-5)

Or use the pipeline with custom settings:

```bash
python run_pipeline.py \
    --model-name roberta-base \
    --num-epochs 10 \
    --batch-size 32 \
    --learning-rate 3e-5
```

### Inference

#### Single Text Detection

```python
from src.inference import PIIDetector

detector = PIIDetector(
    model_path="./models/best_model",
    confidence_threshold=0.5
)

result = detector.detect("Contact John at john@email.com")
```

Returns:
```python
{
    "has_pii": True,
    "pii_types": ["PERSON", "EMAIL"],
    "entities": [
        {"text": "John", "type": "PERSON", "confidence": 0.95},
        {"text": "john@email.com", "type": "EMAIL", "confidence": 0.98}
    ],
    "message": "PII detected in input text"
}
```

Or `None` if no PII detected.

#### Batch Processing

```python
texts = [
    "My name is Alice",
    "The weather is nice",
    "Call me at 555-0100"
]

results = detector.batch_detect(texts)
stats = detector.get_pii_statistics(results)
```

#### Fast Batch Processing

```python
from src.inference import FastPIIDetector

detector = FastPIIDetector(
    model_path="./models/best_model",
    batch_size=32
)

results = detector.batch_detect_optimized(large_text_list)
```

## REST API

### Start API Server

```bash
python src/api.py --model-path ./models/best_model --port 5000
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Single Text Detection

```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Email me at user@example.com"}'
```

Response:
```json
{
  "has_pii": true,
  "pii_types": ["EMAIL"],
  "entities": [
    {
      "text": "user@example.com",
      "type": "EMAIL",
      "confidence": 0.97
    }
  ],
  "message": "PII detected in input text"
}
```

#### 3. Batch Detection

```bash
curl -X POST http://localhost:5000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "My name is John",
      "The sky is blue",
      "SSN: 123-45-6789"
    ]
  }'
```

Response:
```json
{
  "results": [
    {"has_pii": true, ...},
    null,
    {"has_pii": true, ...}
  ],
  "statistics": {
    "total_texts": 3,
    "texts_with_pii": 2,
    "texts_without_pii": 1,
    "pii_rate": 0.67,
    "pii_type_distribution": {
      "PERSON": 1,
      "SSN": 1
    }
  }
}
```

#### 4. Model Info

```bash
curl http://localhost:5000/info
```

## Docker Deployment

### Build Image

```bash
docker build -t pii-detector .
```

### Run Container

```bash
docker run -p 5000:5000 pii-detector
```

### With Custom Model

```bash
docker run -p 5000:5000 \
  -v /path/to/model:/app/models/best_model \
  pii-detector
```

## Supported PII Types

- **PERSON**: Person names
- **EMAIL**: Email addresses
- **PHONE**: Phone numbers
- **SSN**: Social Security Numbers
- **CC**: Credit card numbers
- **ADDRESS**: Physical addresses
- **DATE**: Dates of birth
- **ORG**: Organizations
- **LOC**: Locations

## Configuration

### Model Selection

Choose base model based on requirements:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| bert-base-cased | 110M | Fast | Good |
| roberta-base | 125M | Medium | Better |
| deberta-v3-base | 184M | Slow | Best |

### Confidence Threshold

Adjust sensitivity:

```python
detector = PIIDetector(
    model_path="./models/best_model",
    confidence_threshold=0.7  # Higher = more conservative
)
```

## Performance

Expected metrics on test set:

- Precision: 90-95%
- Recall: 85-92%
- F1 Score: 88-93%

Inference speed (single text):
- CPU: ~50-100ms
- GPU: ~10-20ms

Batch processing (32 texts):
- CPU: ~500ms
- GPU: ~100ms

## Production Deployment

### Gunicorn (Recommended)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.api:app
```

### System Service

Create `/etc/systemd/system/pii-detector.service`:

```ini
[Unit]
Description=PII Detection API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/pii-detector
Environment="MODEL_PATH=/opt/pii-detector/models/best_model"
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:5000 src.api:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl start pii-detector
sudo systemctl enable pii-detector
```

## Monitoring

Add logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

Track metrics:
- Detection latency
- PII detection rate
- False positive/negative rates
- API request volume

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python src/train.py --batch-size 8
```

### Slow Inference

Use FastPIIDetector with GPU:
```python
from src.inference import FastPIIDetector
detector = FastPIIDetector(model_path="./models/best_model", batch_size=64)
```

### Low Accuracy

- Increase training epochs
- Use larger base model (RoBERTa, DeBERTa)
- Add domain-specific training data
- Adjust confidence threshold

## Research References

1. Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
2. Liu et al. (2019) - RoBERTa: A Robustly Optimized BERT Pretraining Approach
3. Dernoncourt et al. (2017) - De-identification of Patient Notes with Recurrent Neural Networks
4. Lample et al. (2016) - Neural Architectures for Named Entity Recognition

## License

MIT License
