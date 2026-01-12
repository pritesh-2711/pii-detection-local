import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

class PIIDetector:
    """
    Production PII detector for identifying PII in text.
    Returns whether PII exists without masking.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize PII detector.
        
        Args:
            model_path: Path to fine-tuned model directory
            confidence_threshold: Minimum confidence for PII detection (0-1)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Load label mappings
        with open(self.model_path / "label_mapping.json", 'r') as f:
            label_info = json.load(f)
        
        self.label2id = label_info['label2id']
        self.id2label = {int(k): v for k, v in label_info['id2label'].items()}
        
        # Load model and tokenizer
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        
        # PII entity types (excluding 'O')
        self.pii_types = [label for label in self.id2label.values() if label != "O"]
    
    def detect(self, text: str) -> Optional[Dict]:
        """
        Detect if PII exists in the input text.
        
        Args:
            text: Input text to check for PII
            
        Returns:
            Dictionary with PII detection results if found, None otherwise
            Format: {
                "has_pii": True,
                "pii_types": ["PERSON", "EMAIL"],
                "entities": [{"text": "John", "type": "PERSON", "confidence": 0.95}, ...],
                "message": "PII detected in input text"
            }
        """
        if not text or not text.strip():
            return None
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Process predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        pred_labels = torch.argmax(predictions, dim=-1)[0].cpu().numpy()
        confidences = torch.max(predictions, dim=-1)[0][0].cpu().numpy()
        
        # Extract entities
        entities = self._extract_entities(tokens, pred_labels, confidences, text)
        
        # Check if any PII was detected
        if entities:
            pii_types = list(set([e['type'] for e in entities]))
            
            return {
                "has_pii": True,
                "pii_types": sorted(pii_types),
                "entities": entities,
                "message": "PII detected in input text"
            }
        
        return None
    
    def _extract_entities(
        self,
        tokens: List[str],
        labels: np.ndarray,
        confidences: np.ndarray,
        original_text: str
    ) -> List[Dict]:
        """
        Extract entities from token predictions.
        """
        entities = []
        current_entity = None
        current_tokens = []
        current_confidences = []
        
        for token, label_id, confidence in zip(tokens, labels, confidences):
            label = self.id2label[label_id]
            
            # Skip special tokens and low confidence
            if token in ['[CLS]', '[SEP]', '<s>', '</s>', '[PAD]']:
                continue
            
            if confidence < self.confidence_threshold:
                continue
            
            # Handle BIO tagging
            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entity_text = self._reconstruct_text(current_tokens)
                    avg_confidence = np.mean(current_confidences)
                    entities.append({
                        "text": entity_text,
                        "type": current_entity,
                        "confidence": float(avg_confidence)
                    })
                
                # Start new entity
                current_entity = label[2:]  # Remove "B-"
                current_tokens = [token]
                current_confidences = [confidence]
            
            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
                current_confidences.append(confidence)
            
            else:
                # End current entity
                if current_entity:
                    entity_text = self._reconstruct_text(current_tokens)
                    avg_confidence = np.mean(current_confidences)
                    entities.append({
                        "text": entity_text,
                        "type": current_entity,
                        "confidence": float(avg_confidence)
                    })
                    current_entity = None
                    current_tokens = []
                    current_confidences = []
        
        # Handle last entity
        if current_entity:
            entity_text = self._reconstruct_text(current_tokens)
            avg_confidence = np.mean(current_confidences)
            entities.append({
                "text": entity_text,
                "type": current_entity,
                "confidence": float(avg_confidence)
            })
        
        return entities
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """
        Reconstruct text from subword tokens.
        """
        text = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]
            elif token.startswith("Ġ"):
                text += " " + token[1:]
            else:
                if text:
                    text += " "
                text += token
        
        return text.strip()
    
    def batch_detect(self, texts: List[str]) -> List[Optional[Dict]]:
        """
        Detect PII in multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of detection results (None if no PII found)
        """
        return [self.detect(text) for text in texts]
    
    def get_pii_statistics(self, results: List[Optional[Dict]]) -> Dict:
        """
        Get statistics from batch detection results.
        
        Args:
            results: List of detection results
            
        Returns:
            Dictionary with statistics
        """
        total = len(results)
        with_pii = sum(1 for r in results if r is not None)
        
        # Count PII types
        pii_type_counts = {}
        for result in results:
            if result:
                for pii_type in result['pii_types']:
                    pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1
        
        return {
            "total_texts": total,
            "texts_with_pii": with_pii,
            "texts_without_pii": total - with_pii,
            "pii_rate": with_pii / total if total > 0 else 0,
            "pii_type_distribution": pii_type_counts
        }

class FastPIIDetector(PIIDetector):
    """
    Optimized version for production with batching and caching.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, batch_size: int = 32):
        super().__init__(model_path, confidence_threshold)
        self.batch_size = batch_size
    
    @torch.inference_mode()
    def batch_detect_optimized(self, texts: List[str]) -> List[Optional[Dict]]:
        """
        Optimized batch detection with proper batching.
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            
            # Process each text in batch
            for idx in range(len(batch_texts)):
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][idx])
                pred_labels = torch.argmax(predictions[idx], dim=-1).cpu().numpy()
                confidences = torch.max(predictions[idx], dim=-1)[0].cpu().numpy()
                
                entities = self._extract_entities(tokens, pred_labels, confidences, batch_texts[idx])
                
                if entities:
                    pii_types = list(set([e['type'] for e in entities]))
                    results.append({
                        "has_pii": True,
                        "pii_types": sorted(pii_types),
                        "entities": entities,
                        "message": "PII detected in input text"
                    })
                else:
                    results.append(None)
        
        return results

def main():
    """
    Example usage.
    """
    # Initialize detector
    detector = PIIDetector(model_path="./models/best_model")
    
    # Test cases
    test_texts = [
        "My name is John Doe and my email is john.doe@example.com",
        "Call me at 555-123-4567 or email jane@company.org",
        "The weather is nice today.",
        "SSN: 123-45-6789, DOB: 01/15/1990",
        "Send the package to 123 Main St, New York, NY 10001",
    ]
    
    print("=== PII Detection Results ===\n")
    
    for i, text in enumerate(test_texts, 1):
        result = detector.detect(text)
        
        print(f"Text {i}: {text}")
        
        if result:
            print(f"  Status: PII DETECTED")
            print(f"  Types: {', '.join(result['pii_types'])}")
            print(f"  Entities:")
            for entity in result['entities']:
                print(f"    - {entity['text']} ({entity['type']}, confidence: {entity['confidence']:.2f})")
        else:
            print(f"  Status: No PII detected")
        
        print()

if __name__ == "__main__":
    main()
