#!/usr/bin/env python3
"""
Test suite for PII detection model.
"""

import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent / "src"))

from inference import PIIDetector

def test_pii_detector():
    """
    Run comprehensive tests on the PII detector.
    """
    print("="*80)
    print("PII DETECTOR TEST SUITE")
    print("="*80)
    
    # Initialize detector
    print("\n[1] Initializing detector...")
    detector = PIIDetector(model_path="./models/best_model")
    print("Detector loaded successfully")
    
    # Test cases
    test_cases = [
        {
            "name": "Email Detection",
            "text": "Contact me at john.doe@example.com",
            "expected_pii": True,
            "expected_types": ["EMAIL"]
        },
        {
            "name": "Person Name Detection",
            "text": "My name is Sarah Johnson and I live in California",
            "expected_pii": True,
            "expected_types": ["PERSON"]
        },
        {
            "name": "Phone Number Detection",
            "text": "Call me at 555-123-4567",
            "expected_pii": True,
            "expected_types": ["PHONE"]
        },
        {
            "name": "Multiple PII Types",
            "text": "John Smith, email: jsmith@company.com, phone: 555-0100",
            "expected_pii": True,
            "expected_types": ["PERSON", "EMAIL", "PHONE"]
        },
        {
            "name": "SSN Detection",
            "text": "My SSN is 123-45-6789",
            "expected_pii": True,
            "expected_types": ["SSN"]
        },
        {
            "name": "Address Detection",
            "text": "Ship to 123 Main Street, New York, NY 10001",
            "expected_pii": True,
            "expected_types": ["ADDRESS"]
        },
        {
            "name": "No PII",
            "text": "The weather is nice today and I enjoy programming",
            "expected_pii": False,
            "expected_types": []
        },
        {
            "name": "Empty String",
            "text": "",
            "expected_pii": False,
            "expected_types": []
        },
        {
            "name": "Organization Detection",
            "text": "I work at Google and Microsoft",
            "expected_pii": True,
            "expected_types": ["ORG"]
        },
        {
            "name": "Date of Birth",
            "text": "Born on 01/15/1990",
            "expected_pii": True,
            "expected_types": ["DATE"]
        }
    ]
    
    # Run tests
    print("\n[2] Running test cases...")
    print("-"*80)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Input: {test['text'][:50]}{'...' if len(test['text']) > 50 else ''}")
        
        result = detector.detect(test['text'])
        
        # Check if PII was detected as expected
        has_pii = result is not None
        
        if has_pii == test['expected_pii']:
            print(f"  PII Detection: PASS (expected: {test['expected_pii']}, got: {has_pii})")
            
            if has_pii:
                detected_types = set(result['pii_types'])
                expected_types = set(test['expected_types'])
                
                # Check if at least some expected types were found
                if detected_types & expected_types:
                    print(f"  PII Types: PASS")
                    print(f"    Expected: {sorted(expected_types)}")
                    print(f"    Detected: {sorted(detected_types)}")
                    print(f"    Entities: {result['entities']}")
                    passed += 1
                else:
                    print(f"  PII Types: FAIL")
                    print(f"    Expected: {sorted(expected_types)}")
                    print(f"    Detected: {sorted(detected_types)}")
                    failed += 1
            else:
                passed += 1
        else:
            print(f"  PII Detection: FAIL (expected: {test['expected_pii']}, got: {has_pii})")
            if result:
                print(f"    Unexpected detection: {result}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_cases)*100):.1f}%")
    
    # Batch test
    print("\n[3] Testing batch processing...")
    batch_texts = [tc['text'] for tc in test_cases[:5]]
    batch_results = detector.batch_detect(batch_texts)
    print(f"Batch processing: {len(batch_results)} results returned")
    
    # Statistics test
    stats = detector.get_pii_statistics(batch_results)
    print(f"\nBatch statistics:")
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_pii_detector()
