from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from inference import PIIDetector, FastPIIDetector

app = Flask(__name__)
CORS(app)

# Global detector instance
detector = None

def initialize_detector(model_path: str, use_fast: bool = True):
    """
    Initialize the PII detector on app startup.
    """
    global detector
    
    if use_fast:
        detector = FastPIIDetector(model_path=model_path, confidence_threshold=0.5, batch_size=32)
    else:
        detector = PIIDetector(model_path=model_path, confidence_threshold=0.5)
    
    print(f"PII Detector initialized with model from {model_path}")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None
    }), 200

@app.route('/detect', methods=['POST'])
def detect_pii():
    """
    Detect PII in input text.
    
    Request body:
    {
        "text": "My email is john@example.com"
    }
    
    Response:
    {
        "has_pii": true,
        "pii_types": ["EMAIL"],
        "entities": [{"text": "john@example.com", "type": "EMAIL", "confidence": 0.95}],
        "message": "PII detected in input text"
    }
    or None if no PII detected
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not isinstance(text, str):
            return jsonify({
                "error": "'text' must be a string"
            }), 400
        
        if not text.strip():
            return jsonify({
                "error": "Empty text provided"
            }), 400
        
        # Detect PII
        result = detector.detect(text)
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify(None), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/detect/batch', methods=['POST'])
def detect_pii_batch():
    """
    Detect PII in multiple texts.
    
    Request body:
    {
        "texts": ["Text 1", "Text 2", ...]
    }
    
    Response:
    {
        "results": [result1, result2, ...],
        "statistics": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                "error": "'texts' must be a list"
            }), 400
        
        if not texts:
            return jsonify({
                "error": "Empty texts list provided"
            }), 400
        
        # Detect PII in batch
        if isinstance(detector, FastPIIDetector):
            results = detector.batch_detect_optimized(texts)
        else:
            results = detector.batch_detect(texts)
        
        # Get statistics
        stats = detector.get_pii_statistics(results)
        
        return jsonify({
            "results": results,
            "statistics": stats
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    """
    if detector is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    return jsonify({
        "model_path": str(detector.model_path),
        "device": str(detector.device),
        "confidence_threshold": detector.confidence_threshold,
        "supported_pii_types": detector.pii_types,
        "num_labels": len(detector.id2label)
    }), 200

def create_app(model_path: str = "./models/best_model", use_fast: bool = True):
    """
    Application factory.
    """
    initialize_detector(model_path, use_fast)
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PII Detection API Server')
    parser.add_argument('--model-path', type=str, default='./models/best_model',
                       help='Path to trained model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--use-fast', action='store_true', default=True,
                       help='Use optimized fast detector')
    
    args = parser.parse_args()
    
    # Initialize detector
    initialize_detector(args.model_path, args.use_fast)
    
    # Run app
    print(f"\nStarting PII Detection API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
