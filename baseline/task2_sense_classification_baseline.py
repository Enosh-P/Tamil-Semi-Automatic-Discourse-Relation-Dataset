#!/usr/bin/env python3
"""
Task 2: Sense Classification Baseline
Majority Sense Baseline - predicts the most frequent sense for each connective.
"""

import json
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split


class SenseClassificationBaseline:
    """
    Majority Sense Baseline for discourse relation sense classification.
    For each connective, predicts its most frequent sense from training data.
    """
    
    def __init__(self):
        """Initialize the baseline."""
        self.connective_sense_map: Dict[str, str] = {}
        self.connective_sense_counts: Dict[str, Counter] = defaultdict(Counter)
        self.global_majority_sense: str = ""
        self.all_senses: Counter = Counter()
        
    def train(self, train_data: List[Dict]):
        """
        Build the connective-to-sense mapping from training data.
        
        Args:
            train_data: List of training examples (JSON format)
        """
        print("Training majority sense baseline...")
        
        for item in train_data:
            tamil_data = item.get("tamil", {})
            connective = tamil_data.get("connective", {}).get("raw_text", "")
            sense = tamil_data.get("relation", {}).get("sense", "")
            
            if connective and sense:
                connective = connective.strip()
                sense = sense.strip()
                
                # Count sense for this connective
                self.connective_sense_counts[connective][sense] += 1
                
                # Count global sense distribution
                self.all_senses[sense] += 1
        
        # Determine majority sense for each connective
        for connective, sense_counts in self.connective_sense_counts.items():
            # Get the most common sense for this connective
            majority_sense = sense_counts.most_common(1)[0][0]
            self.connective_sense_map[connective] = majority_sense
        
        # Determine global majority sense (fallback for unseen connectives)
        if self.all_senses:
            self.global_majority_sense = self.all_senses.most_common(1)[0][0]
        
        print(f"  Trained on {len(self.connective_sense_map)} unique connectives")
        print(f"  Global majority sense: {self.global_majority_sense}")
    
    def predict(self, connective: str) -> str:
        """
        Predict the sense for a given connective.
        
        Args:
            connective: Tamil connective text
        
        Returns:
            Predicted sense label
        """
        connective = connective.strip() if connective else ""
        
        # If connective seen in training, return its majority sense
        if connective in self.connective_sense_map:
            return self.connective_sense_map[connective]
        
        # If unseen, return global majority sense
        return self.global_majority_sense
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate the baseline on test data.
        
        Args:
            test_data: List of test examples
        
        Returns:
            Dictionary with evaluation metrics
        """
        total = 0
        correct = 0
        seen_correct = 0
        seen_total = 0
        unseen_correct = 0
        unseen_total = 0
        
        sense_confusion = defaultdict(lambda: defaultdict(int))
        
        for item in test_data:
            tamil_data = item.get("tamil", {})
            connective = tamil_data.get("connective", {}).get("raw_text", "")
            gold_sense = tamil_data.get("relation", {}).get("sense", "")
            
            if not connective or not gold_sense:
                continue
            
            connective = connective.strip()
            gold_sense = gold_sense.strip()
            
            # Predict
            predicted_sense = self.predict(connective)
            
            total += 1
            
            # Track confusion matrix
            sense_confusion[gold_sense][predicted_sense] += 1
            
            # Check if correct
            if predicted_sense == gold_sense:
                correct += 1
            
            # Track seen vs unseen performance
            if connective in self.connective_sense_map:
                seen_total += 1
                if predicted_sense == gold_sense:
                    seen_correct += 1
            else:
                unseen_total += 1
                if predicted_sense == gold_sense:
                    unseen_correct += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        seen_accuracy = seen_correct / seen_total if seen_total > 0 else 0
        unseen_accuracy = unseen_correct / unseen_total if unseen_total > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "seen_total": seen_total,
            "seen_correct": seen_correct,
            "seen_accuracy": seen_accuracy,
            "unseen_total": unseen_total,
            "unseen_correct": unseen_correct,
            "unseen_accuracy": unseen_accuracy,
            "sense_confusion": dict(sense_confusion)
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the trained model."""
        stats = {
            "num_connectives": len(self.connective_sense_map),
            "num_senses": len(self.all_senses),
            "global_majority_sense": self.global_majority_sense,
            "sense_distribution": dict(self.all_senses),
            "connective_examples": {}
        }
        
        # Get top 10 connectives by frequency
        connective_freq = {
            conn: sum(counts.values())
            for conn, counts in self.connective_sense_counts.items()
        }
        top_connectives = sorted(connective_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for conn, freq in top_connectives:
            majority_sense = self.connective_sense_map[conn]
            sense_dist = dict(self.connective_sense_counts[conn])
            stats["connective_examples"][conn] = {
                "frequency": freq,
                "majority_sense": majority_sense,
                "sense_distribution": sense_dist
            }
        
        return stats


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


def load_connective_sense_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file containing connective-sense mappings.
    This can be used to verify the baseline's learned mappings.
    """
    print(f"\nLoading connective-sense CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} connective-sense pairs")
    return df


def main():
    """Main function to run Task 2 baseline."""
    print("=" * 70)
    print("Task 2: Sense Classification Baseline")
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
    
    # Initialize and train baseline
    print("\n[3] Training baseline...")
    baseline = SenseClassificationBaseline()
    baseline.train(train_data)
    
    # Get training statistics
    print("\n[4] Training Statistics:")
    stats = baseline.get_statistics()
    print(f"  Number of unique connectives: {stats['num_connectives']}")
    print(f"  Number of unique senses: {stats['num_senses']}")
    print(f"  Global majority sense: {stats['global_majority_sense']}")
    
    print(f"\n  Sense distribution:")
    for sense, count in sorted(stats['sense_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sum(stats['sense_distribution'].values())) * 100
        print(f"    {sense}: {count} ({percentage:.1f}%)")
    
    print(f"\n  Top 10 connectives by frequency:")
    for conn, info in stats['connective_examples'].items():
        print(f"    {conn}: {info['frequency']} occurrences → {info['majority_sense']}")
        # Show sense distribution for this connective
        for sense, count in sorted(info['sense_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / info['frequency']) * 100
            print(f"      - {sense}: {count} ({percentage:.1f}%)")
    
    # Evaluate on test set
    print("\n[5] Evaluating on test set...")
    results = baseline.evaluate(test_data)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total test samples: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    print(f"\n--- Performance by Connective Type ---")
    print(f"Seen connectives (in training):")
    print(f"  Count: {results['seen_total']}")
    print(f"  Correct: {results['seen_correct']}")
    print(f"  Accuracy: {results['seen_accuracy']:.4f} ({results['seen_accuracy']*100:.2f}%)")
    
    print(f"\nUnseen connectives (not in training):")
    print(f"  Count: {results['unseen_total']}")
    print(f"  Correct: {results['unseen_correct']}")
    print(f"  Accuracy: {results['unseen_accuracy']:.4f} ({results['unseen_accuracy']*100:.2f}%)")
    
    # Print confusion matrix for top senses
    print(f"\n--- Sense Confusion Matrix (Top Senses) ---")
    confusion = results['sense_confusion']
    all_senses = set()
    for gold in confusion:
        all_senses.add(gold)
        all_senses.update(confusion[gold].keys())
    
    # Get top 5 most frequent senses
    sense_totals = {}
    for gold in confusion:
        sense_totals[gold] = sum(confusion[gold].values())
    top_senses = sorted(sense_totals.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for gold_sense, total in top_senses:
        print(f"\nGold: {gold_sense} (n={total})")
        if gold_sense in confusion:
            for pred_sense, count in sorted(confusion[gold_sense].items(), key=lambda x: x[1], reverse=True)[:3]:
                percentage = (count / total) * 100
                print(f"  → Predicted as {pred_sense}: {count} ({percentage:.1f}%)")
    
    # Test on a few examples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    for i, item in enumerate(test_data[:10]):
        tamil_data = item.get("tamil", {})
        connective = tamil_data.get("connective", {}).get("raw_text", "")
        gold_sense = tamil_data.get("relation", {}).get("sense", "")
        
        if not connective:
            continue
        
        predicted_sense = baseline.predict(connective)
        is_correct = "✓" if predicted_sense == gold_sense else "✗"
        is_seen = "SEEN" if connective in baseline.connective_sense_map else "UNSEEN"
        
        print(f"\nExample {i+1}: {is_correct}")
        print(f"  Connective: {connective} [{is_seen}]")
        print(f"  Gold sense: {gold_sense}")
        print(f"  Predicted:  {predicted_sense}")
    
    print("\n" + "=" * 70)
    
    # Optional: Load and compare with provided CSV
    try:
        print("\n[6] Comparing with provided CSV mappings...")
        connectives_df = load_connective_sense_csv(CONNECTIVES_CSV)
        print("\nNote: This is for verification only. The baseline learns from the training data.")
    except FileNotFoundError:
        print("\n[6] CSV files not found - skipping comparison")


if __name__ == "__main__":
    main()
