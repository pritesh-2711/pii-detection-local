import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class SimplePIIDataCollector:
    """
    Simplified data collector that only uses synthetic data.
    Use this if CoNLL-2003 loading fails.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.pii_labels = [
            "O",
            "B-PERSON", "I-PERSON",
            "B-EMAIL", "I-EMAIL",
            "B-PHONE", "I-PHONE",
            "B-SSN", "I-SSN",
            "B-CC", "I-CC",
            "B-ADDRESS", "I-ADDRESS",
            "B-DATE", "I-DATE",
            "B-ORG", "I-ORG",
            "B-LOC", "I-LOC"
        ]
    
    def generate_synthetic_pii(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic PII examples.
        """
        print(f"Generating {n_samples} synthetic PII examples...")
        
        from faker import Faker
        import random
        
        fake = Faker()
        data = []
        
        # Expanded templates for more variety
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
            "Call {name} at {phone} for more information.",
            "Email {email} to reach {org} customer service.",
            "{name} lives in {loc} and works at {org}.",
            "The meeting is on {date} at {address}.",
            "Contact {org} at {phone} or visit {address}.",
            "Send payment to {name} using card {cc}.",
            "Patient {name} visited on {date}.",
            "{name} from {loc} joined {org} recently.",
            "My phone number is {phone}.",
            "You can email me at {email}.",
            "I was born on {dob}.",
            "The company {org} is located in {loc}.",
            "{name} works as a developer.",
            "Call customer support at {phone}.",
            "My address is {address}.",
            "Social Security Number: {ssn}",
            "Meeting with {name} and {org} representatives.",
            "Card ending in {cc}.",
            "Visit us at {address} in {loc}.",
        ]
        
        for _ in range(n_samples):
            template = random.choice(templates)
            
            pii_values = {
                'email': fake.email(),
                'phone': fake.phone_number(),
                'ssn': fake.ssn(),
                'cc': fake.credit_card_number(),
                'dob': fake.date_of_birth().strftime('%m/%d/%Y'),
                'date': fake.date(),
                'address': fake.address().replace('\n', ', '),
                'name': fake.name(),
                'org': fake.company(),
                'loc': fake.city()
            }
            
            text = template
            entities = []
            
            for key, value in pii_values.items():
                if '{' + key + '}' in text:
                    start = text.find('{' + key + '}')
                    text = text.replace('{' + key + '}', value, 1)
                    
                    label_map = {
                        'email': 'EMAIL',
                        'phone': 'PHONE',
                        'ssn': 'SSN',
                        'cc': 'CC',
                        'dob': 'DATE',
                        'date': 'DATE',
                        'address': 'ADDRESS',
                        'name': 'PERSON',
                        'org': 'ORG',
                        'loc': 'LOC'
                    }
                    
                    entities.append({
                        'start': start,
                        'end': start + len(value),
                        'label': label_map[key]
                    })
            
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
        Tokenize text and assign BIO labels.
        """
        tokens = text.split()
        labels = ["O"] * len(tokens)
        
        char_to_token = {}
        char_pos = 0
        for token_idx, token in enumerate(tokens):
            token_start = text.find(token, char_pos)
            token_end = token_start + len(token)
            for i in range(token_start, token_end):
                char_to_token[i] = token_idx
            char_pos = token_end
        
        for entity in entities:
            start_token = char_to_token.get(entity['start'])
            end_token = char_to_token.get(entity['end'] - 1)
            
            if start_token is not None and end_token is not None:
                labels[start_token] = f"B-{entity['label']}"
                for i in range(start_token + 1, end_token + 1):
                    if i < len(labels):
                        labels[i] = f"I-{entity['label']}"
        
        return tokens, labels
    
    def prepare_dataset(self, n_samples: int = 10000) -> Dict[str, pd.DataFrame]:
        """
        Generate and split dataset.
        """
        print("\n=== Preparing Synthetic Dataset ===")
        
        # Generate synthetic data
        all_data = self.generate_synthetic_pii(n_samples=n_samples)
        
        # Split: 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(all_data, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
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
    collector = SimplePIIDataCollector(data_dir="./data")
    datasets = collector.prepare_dataset(n_samples=10000)