import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class PIIDataCollector:
    """
    Collects and prepares PII datasets for training.
    Uses public datasets: CoNLL-2003, WikiANN, and synthetic PII data.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # PII entity labels (BIO format)
        self.pii_labels = [
            "O",           # Outside
            "B-PERSON",    # Beginning of person name
            "I-PERSON",    # Inside person name
            "B-EMAIL",     # Email address
            "I-EMAIL",
            "B-PHONE",     # Phone number
            "I-PHONE",
            "B-SSN",       # Social Security Number
            "I-SSN",
            "B-CC",        # Credit Card
            "I-CC",
            "B-ADDRESS",   # Physical address
            "I-ADDRESS",
            "B-DATE",      # Date of birth
            "I-DATE",
            "B-ORG",       # Organization
            "I-ORG",
            "B-LOC",       # Location
            "I-LOC"
        ]
        
    def download_conll2003(self) -> pd.DataFrame:
        """
        Download CoNLL-2003 dataset (contains PERSON, ORG, LOC).
        """
        print("Downloading CoNLL-2003 dataset...")
        
        try:
            dataset = load_dataset("conll2003")
            
            # Convert to our format
            data = []
            for split in ['train', 'validation', 'test']:
                for example in dataset[split]:
                    tokens = example['tokens']
                    ner_tags = example['ner_tags']
                    
                    # Convert CoNLL tags to our PII tags
                    labels = self._convert_conll_tags(ner_tags)
                    
                    data.append({
                        'tokens': tokens,
                        'labels': labels,
                        'split': split
                    })
            
            df = pd.DataFrame(data)
            print(f"CoNLL-2003 loaded: {len(df)} examples")
            return df
            
        except Exception as e:
            print(f"Error loading CoNLL-2003: {e}")
            return pd.DataFrame()
    
    def _convert_conll_tags(self, ner_tags: List[int]) -> List[str]:
        """
        Convert CoNLL-2003 NER tags to our PII label scheme.
        CoNLL tags: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
        """
        tag_map = {
            0: "O",
            1: "B-PERSON",
            2: "I-PERSON",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "O",  # MISC -> O
            8: "O"
        }
        return [tag_map.get(tag, "O") for tag in ner_tags]
    
    def generate_synthetic_pii(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic PII examples for email, phone, SSN, credit card, dates.
        """
        print(f"Generating {n_samples} synthetic PII examples...")
        
        from faker import Faker
        import random
        import re
        
        fake = Faker()
        data = []
        
        templates = [
            "My email is {email} and phone is {phone}.",
            "Contact me at {email} or call {phone}.",
            "SSN: {ssn}, DOB: {dob}",
            "Card number {cc} expires on {date}.",
            "I live at {address}.",
            "My name is {name} and I work at {org}.",
            "{name} can be reached at {email} or {phone}.",
            "Patient: {name}, DOB: {dob}, SSN: {ssn}",
            "Shipping to {address} for {name}.",
            "Credit card {cc} belongs to {name}.",
        ]
        
        for _ in range(n_samples):
            template = random.choice(templates)
            
            # Generate PII values
            pii_values = {
                'email': fake.email(),
                'phone': fake.phone_number(),
                'ssn': fake.ssn(),
                'cc': fake.credit_card_number(),
                'dob': fake.date_of_birth().strftime('%m/%d/%Y'),
                'date': fake.date(),
                'address': fake.address().replace('\n', ', '),
                'name': fake.name(),
                'org': fake.company()
            }
            
            # Fill template
            text = template
            entities = []
            
            for key, value in pii_values.items():
                if '{' + key + '}' in text:
                    start = text.find('{' + key + '}')
                    text = text.replace('{' + key + '}', value, 1)
                    
                    # Map key to label type
                    label_map = {
                        'email': 'EMAIL',
                        'phone': 'PHONE',
                        'ssn': 'SSN',
                        'cc': 'CC',
                        'dob': 'DATE',
                        'date': 'DATE',
                        'address': 'ADDRESS',
                        'name': 'PERSON',
                        'org': 'ORG'
                    }
                    
                    entities.append({
                        'start': start,
                        'end': start + len(value),
                        'label': label_map[key]
                    })
            
            # Tokenize and create BIO labels
            tokens, labels = self._tokenize_and_label(text, entities)
            
            data.append({
                'tokens': tokens,
                'labels': labels,
                'split': 'train'
            })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} synthetic examples")
        return df
    
    def _tokenize_and_label(self, text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Tokenize text and assign BIO labels based on entity positions.
        """
        # Simple whitespace tokenization
        tokens = text.split()
        labels = ["O"] * len(tokens)
        
        # Track character positions
        char_to_token = {}
        char_pos = 0
        for token_idx, token in enumerate(tokens):
            token_start = text.find(token, char_pos)
            token_end = token_start + len(token)
            for i in range(token_start, token_end):
                char_to_token[i] = token_idx
            char_pos = token_end
        
        # Assign labels based on entities
        for entity in entities:
            start_token = char_to_token.get(entity['start'])
            end_token = char_to_token.get(entity['end'] - 1)
            
            if start_token is not None and end_token is not None:
                labels[start_token] = f"B-{entity['label']}"
                for i in range(start_token + 1, end_token + 1):
                    labels[i] = f"I-{entity['label']}"
        
        return tokens, labels
    
    def prepare_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Combine all datasets and split into train/val/test.
        """
        print("\n=== Preparing Full Dataset ===")
        
        # Collect data from multiple sources
        conll_df = self.download_conll2003()
        synthetic_df = self.generate_synthetic_pii(n_samples=5000)
        
        # Combine
        all_data = pd.concat([conll_df, synthetic_df], ignore_index=True)
        
        # Split by existing splits or create new ones
        train_df = all_data[all_data['split'] == 'train'].reset_index(drop=True)
        val_df = all_data[all_data['split'] == 'validation'].reset_index(drop=True)
        test_df = all_data[all_data['split'] == 'test'].reset_index(drop=True)
        
        # If validation is empty, create from train
        if len(val_df) == 0:
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        # If test is empty, create from train
        if len(test_df) == 0:
            train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_df)}")
        print(f"  Validation: {len(val_df)}")
        print(f"  Test: {len(test_df)}")
        
        # Save to disk
        train_df.to_json(self.data_dir / "train.jsonl", orient='records', lines=True)
        val_df.to_json(self.data_dir / "val.jsonl", orient='records', lines=True)
        test_df.to_json(self.data_dir / "test.jsonl", orient='records', lines=True)
        
        # Save label mapping
        label2id = {label: idx for idx, label in enumerate(self.pii_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        with open(self.data_dir / "label_mapping.json", 'w') as f:
            json.dump({
                'label2id': label2id,
                'id2label': id2label,
                'labels': self.pii_labels
            }, f, indent=2)
        
        print(f"\nDatasets saved to {self.data_dir}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

if __name__ == "__main__":
    collector = PIIDataCollector(data_dir="./data")
    datasets = collector.prepare_dataset()
