#!/usr/bin/env python3
"""
Task 1: Connective Detection Baseline
Dictionary/List-based baseline for detecting discourse connectives in Tamil sentences.
"""

import json
import pandas as pd
from typing import List, Dict, Set, Tuple
from sklearn.model_selection import train_test_split
import re


class ConnectiveDetectionBaseline:
    """
    Dictionary-based baseline for connective detection.
    Uses a pre-built lexicon to detect connectives in sentences.
    """
    
    def __init__(self, connectives_csv: str = None):
        """
        Initialize the baseline with connective lexicons.
        
        Args:
            connectives_csv: Path to CSV containing Tamil connectives
        """
        self.lexical_connectives: Set[str] = set()
        self.suffixal_connectives: Set[str] = set()
        self.all_connectives: Set[str] = set()
        
        if connectives_csv:
            self.load_tamil_connectives(connectives_csv)
    
    def load_tamil_connectives(self, csv_path: str):
        """Load lexical connectives from CSV file."""
        print(f"Loading Tamil connectives from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # All unique Tamil connectives
        self.all_connectives = set(df["tamil_connective"].dropna())

        # Lexical connectives (do NOT start with '-' or '−')
        self.lexical_connectives = {
            c for c in self.all_connectives
            if not str(c).startswith(("-", "−"))
        }

        # Suffixal connectives (start with '-' or '−')
        self.suffixal_connectives = {
            c for c in self.all_connectives
            if str(c).startswith(("-", "−"))
        }
        print(f"  Loaded {len(self.lexical_connectives)} lexical connectives")
        print(f"  Loaded {len(self.suffixal_connectives)} suffixal connectives")
        print(f"  Loaded {len(self.all_connectives)} Total connectives")
    
    def add_connective(self, connective: str, is_suffix: bool = False):
        """Manually add a connective to the lexicon."""
        connective = connective.strip()
        if connective:
            if is_suffix:
                self.suffixal_connectives.add(connective)
            else:
                self.lexical_connectives.add(connective)
            self.all_connectives.add(connective)
    
    def tokenize_tamil(self, sentence: str) -> List[str]:
        """
        Simple tokenizer for Tamil text.
        Splits on whitespace and punctuation.
        """
        # Split on whitespace and common punctuation
        tokens = re.findall(r'\S+', sentence)
        return tokens
    
    def detect_connective(self, sentence: str, return_all: bool = False) -> List[Dict]:
        """
        Detect connectives in a Tamil sentence using dictionary lookup.
        
        Args:
            sentence: Tamil sentence text
            return_all: If True, return all occurrences; if False, return first only
        
        Returns:
            List of detected connectives with their information
        """
        if not sentence:
            return []
        
        tokens = self.tokenize_tamil(sentence)
        detections = []
        
        for i, token in enumerate(tokens):
            # Check exact match
            if token in self.all_connectives:
                connective_type = "lexical" if token in self.lexical_connectives else "suffixal"
                
                # Find the position in the original sentence
                start_pos = sentence.find(token)
                if start_pos != -1:
                    detection = {
                        "connective": token,
                        "type": connective_type,
                        "position": i,
                        "span_start": start_pos,
                        "span_end": start_pos + len(token)
                    }
                    detections.append(detection)
                    
                    if not return_all:
                        return detections
            
            # Also check if any suffix is part of the token (for suffixal connectives)
            for suffix in self.suffixal_connectives:
                if token.endswith(suffix) and suffix != token:
                    start_pos = sentence.find(token)
                    if start_pos != -1:
                        suffix_start = start_pos + len(token) - len(suffix)
                        detection = {
                            "connective": suffix,
                            "type": "suffixal",
                            "position": i,
                            "token": token,
                            "span_start": suffix_start,
                            "span_end": suffix_start + len(suffix)
                        }
                        detections.append(detection)
                        
                        if not return_all:
                            return detections
        
        return detections
    
    def predict(self, sentence: str) -> Dict:
        """
        Predict whether a connective is present in the sentence.
        
        Returns:
            Dictionary with prediction and detected connectives
        """
        detections = self.detect_connective(sentence, return_all=False)
        
        return {
            "has_connective": len(detections) > 0,
            "detections": detections
        }
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate the baseline on test data.
        
        Args:
            test_data: List of JSON objects with Tamil discourse relations
        
        Returns:
            Dictionary with evaluation metrics
        """
        total = len(test_data)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        correct_detections = 0
        
        for item in test_data:
            tamil_data = item.get("tamil", {})
            sentence = tamil_data.get("sentence", "")
            gold_connective = tamil_data.get("connective", {}).get("raw_text", "")
            
            # Predict
            prediction = self.predict(sentence)
            predicted_has = prediction["has_connective"]
            gold_has = bool(gold_connective and gold_connective.strip())
            
            # Calculate metrics
            if predicted_has and gold_has:
                true_positives += 1
                
                # Check if we detected the correct connective
                detected_connectives = [d["connective"] for d in prediction["detections"]]
                if gold_connective in detected_connectives:
                    correct_detections += 1
            elif predicted_has and not gold_has:
                false_positives += 1
            elif not predicted_has and gold_has:
                false_negatives += 1
            else:
                true_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        exact_match_accuracy = correct_detections / true_positives if true_positives > 0 else 0
        
        return {
            "total": total,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "exact_match_accuracy": exact_match_accuracy
        }


def load_dataset(json_path: str) -> List[Dict]:
    """Load the discourse relations dataset from JSON."""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} relations")
    return data


def split_dataset(data: List[Dict], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and test sets.
    
    Args:
        data: List of discourse relations
        test_size: Proportion of test set (default 0.2 = 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data)
    """
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"\nDataset split:")
    print(f"  Training set: {len(train_data)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set: {len(test_data)} samples ({test_size*100:.0f}%)")
    
    return train_data, test_data


def main():
    """Main function to run Task 1 baseline."""
    print("=" * 70)
    print("Task 1: Connective Detection Baseline")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = "../dataset/parallel_dataset.json"
    CONNECTIVES_CSV = "../tamil_connectives.csv"
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    # Split dataset (80% train, 20% test)
    print("\n[2] Splitting dataset...")
    train_data, test_data = split_dataset(dataset, test_size=0.2, random_state=42)
    
    # Initialize baseline
    print("\n[3] Initializing baseline with connective lexicons...")
    baseline = ConnectiveDetectionBaseline(
        connectives_csv=CONNECTIVES_CSV
    )
    
    # Print lexicon statistics
    print(f"\n[4] Lexicon Statistics:")
    print(f"  Total unique connectives: {len(baseline.all_connectives)}")
    print(f"  Lexical connectives: {len(baseline.lexical_connectives)}")
    print(f"  Suffixal connectives: {len(baseline.suffixal_connectives)}")
    
    # Sample connectives
    print(f"\n  Sample lexical connectives: {list(baseline.lexical_connectives)[:10]}")
    print(f"  Sample suffixal connectives: {list(baseline.suffixal_connectives)[:10]}")
    
    # Evaluate on test set
    print("\n[5] Evaluating on test set...")
    results = baseline.evaluate(test_data)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total test samples: {results['total']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {results['true_positives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  True Negatives:  {results['true_negatives']}")
    print(f"  False Negatives: {results['false_negatives']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1 Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"\nExact Match Accuracy: {results['exact_match_accuracy']:.4f} ({results['exact_match_accuracy']*100:.2f}%)")
    print("  (Percentage of correct connective identification among detected positives)")
    
    # Test on a few examples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    for i, item in enumerate(test_data[:5]):
        tamil_data = item.get("tamil", {})
        sentence = tamil_data.get("sentence", "")
        gold_connective = tamil_data.get("connective", {}).get("raw_text", "")
        
        prediction = baseline.predict(sentence)
        
        print(f"\nExample {i+1}:")
        print(f"  Sentence: {sentence[:100]}...")
        print(f"  Gold connective: {gold_connective}")
        print(f"  Prediction: {prediction['has_connective']}")
        if prediction['detections']:
            print(f"  Detected: {[d['connective'] for d in prediction['detections']]}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
