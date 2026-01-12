#!/usr/bin/env python3
"""
Example client for PII Detection API.
"""

import requests
import json

class PIIDetectionClient:
    """
    Client for interacting with PII Detection API.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def detect(self, text: str):
        """Detect PII in single text."""
        response = requests.post(
            f"{self.base_url}/detect",
            json={"text": text}
        )
        return response.json()
    
    def detect_batch(self, texts: list):
        """Detect PII in multiple texts."""
        response = requests.post(
            f"{self.base_url}/detect/batch",
            json={"texts": texts}
        )
        return response.json()
    
    def get_info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/info")
        return response.json()

def main():
    """
    Example usage of PII Detection API client.
    """
    print("=== PII Detection API Client Example ===\n")
    
    # Initialize client
    client = PIIDetectionClient(base_url="http://localhost:5000")
    
    # Health check
    print("[1] Health Check")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}\n")
    
    # Single detection
    print("[2] Single Text Detection")
    test_text = "My name is Alice Johnson and my email is alice@example.com"
    print(f"Text: {test_text}")
    
    result = client.detect(test_text)
    
    if result:
        print(f"Result: PII DETECTED")
        print(f"Types: {', '.join(result['pii_types'])}")
        print(f"Entities:")
        for entity in result['entities']:
            print(f"  - {entity['text']} ({entity['type']}, confidence: {entity['confidence']:.2f})")
    else:
        print("Result: No PII detected")
    
    print()
    
    # Batch detection
    print("[3] Batch Detection")
    test_texts = [
        "Contact support at help@company.com",
        "The weather is sunny today",
        "My SSN is 123-45-6789",
        "Call Bob at 555-0100"
    ]
    
    print(f"Processing {len(test_texts)} texts...")
    batch_result = client.detect_batch(test_texts)
    
    print(f"\nResults:")
    for i, result in enumerate(batch_result['results'], 1):
        print(f"  Text {i}: {'PII Found' if result else 'No PII'}")
        if result:
            print(f"    Types: {', '.join(result['pii_types'])}")
    
    print(f"\nStatistics:")
    stats = batch_result['statistics']
    print(f"  Total texts: {stats['total_texts']}")
    print(f"  With PII: {stats['texts_with_pii']}")
    print(f"  Without PII: {stats['texts_without_pii']}")
    print(f"  PII rate: {stats['pii_rate']:.1%}")
    
    print()
    
    # Model info
    print("[4] Model Information")
    info = client.get_info()
    print(f"Model path: {info['model_path']}")
    print(f"Device: {info['device']}")
    print(f"Confidence threshold: {info['confidence_threshold']}")
    print(f"Supported PII types: {', '.join(info['supported_pii_types'][:5])}...")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the server is running: python src/api.py")
    except Exception as e:
        print(f"Error: {e}")
