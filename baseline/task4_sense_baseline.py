#!/usr/bin/env python3
"""
Task 4: Sense Classification Baseline (Full Data Evaluation)
Uses the entire dataset to learn connective-sense mappings and evaluate on the same data.
No train/test split - predicts the most frequent sense for each connective.
"""

import json
import pandas as pd
from typing import List, Dict
from collections import defaultdict, Counter


class SenseClassificationBaseline:
    """
    Majority Sense Baseline for discourse relation sense classification.
    For each connective, predicts its most frequent sense from the data.
    """
    
    def __init__(self):
        """Initialize the baseline."""
        self.connective_sense_map: Dict[str, str] = {}
        self.connective_sense_counts: Dict[str, Counter] = defaultdict(Counter)
        self.global_majority_sense: str = ""
        self.all_senses: Counter = Counter()
        
    def build_sense_mapping(self, data: List[Dict]):
        """
        Build the connective-to-sense mapping from all data.
        
        Args:
            data: List of all examples (JSON format)
        """
        print("Building connective-sense mapping from entire dataset...")
        
        for item in data:
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
        
        print(f"  Built mapping for {len(self.connective_sense_map)} unique connectives")
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
        
        # If connective exists in mapping, return its majority sense
        if connective in self.connective_sense_map:
            return self.connective_sense_map[connective]
        
        # If not found, return global majority sense
        return self.global_majority_sense
    
    def evaluate(self, data: List[Dict]) -> Dict:
        """
        Evaluate the baseline on the data.
        
        Args:
            data: List of examples
        
        Returns:
            Dictionary with evaluation metrics
        """
        total = 0
        correct = 0
        
        sense_confusion = defaultdict(lambda: defaultdict(int))
        
        for item in data:
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
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "sense_confusion": dict(sense_confusion)
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the learned mappings."""
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


def load_connective_sense_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file containing connective-sense mappings.
    This can be used to verify the baseline's learned mappings.
    """
    print(f"\nLoading connective-sense CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} connective-sense pairs")
    return df


def save_results(results: Dict, stats: Dict, baseline, output_path: str):
    """
    Save evaluation results to a JSON file with comprehensive metrics.
    
    Args:
        results: Evaluation results dictionary
        stats: Statistics dictionary
        baseline: The trained baseline model
        output_path: Path to save the JSON file
    """
    # Build comprehensive results dictionary
    output_data = {
        "task4_sense_classification": {
            "overall_accuracy": results['accuracy'],
            "total_samples": results['total'],
            "correct_predictions": results['correct'],
            "incorrect_predictions": results['total'] - results['correct'],
            "num_senses": stats['num_senses'],
            "global_majority_sense": stats['global_majority_sense'],
            "sense_distribution": stats['sense_distribution'],
            "top_10_connectives": {}
        }
    }
    
    # Add top 10 connectives info
    for conn, info in stats['connective_examples'].items():
        output_data["task4_sense_classification"]["top_10_connectives"][conn] = {
            "frequency": info['frequency'],
            "majority_sense": info['majority_sense'],
            "sense_distribution": info['sense_distribution']
        }
    
    # Calculate precision, recall, F1 for each sense
    per_sense_metrics = {}
    confusion = results['sense_confusion']
    
    # Calculate predicted counts for each sense (for precision)
    predicted_counts = defaultdict(int)
    for gold_sense in confusion:
        for pred_sense, count in confusion[gold_sense].items():
            predicted_counts[pred_sense] += count
    
    for gold_sense in confusion:
        # True Positives: correctly predicted as this sense
        tp = confusion[gold_sense].get(gold_sense, 0)
        
        # False Negatives: should be this sense but predicted as something else
        fn = sum(confusion[gold_sense].values()) - tp
        
        # False Positives: predicted as this sense but was actually something else
        fp = predicted_counts[gold_sense] - tp
        
        # Calculate metrics
        total = sum(confusion[gold_sense].values())
        accuracy = tp / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_sense_metrics[gold_sense] = {
            "total": total,
            "correct": tp,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "predictions": dict(confusion[gold_sense])
        }
    
    output_data["task4_sense_classification"]["per_sense_metrics"] = per_sense_metrics
    
    # Calculate macro and weighted averages
    macro_precision = sum(m['precision'] for m in per_sense_metrics.values()) / len(per_sense_metrics)
    macro_recall = sum(m['recall'] for m in per_sense_metrics.values()) / len(per_sense_metrics)
    macro_f1 = sum(m['f1_score'] for m in per_sense_metrics.values()) / len(per_sense_metrics)
    
    weighted_precision = sum(m['precision'] * m['total'] for m in per_sense_metrics.values()) / results['total']
    weighted_recall = sum(m['recall'] * m['total'] for m in per_sense_metrics.values()) / results['total']
    weighted_f1 = sum(m['f1_score'] * m['total'] for m in per_sense_metrics.values()) / results['total']
    
    output_data["task4_sense_classification"]["macro_metrics"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1
    }
    
    output_data["task4_sense_classification"]["weighted_metrics"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1
    }
    
    # Add confusion matrix
    output_data["task4_sense_classification"]["confusion_matrix"] = results['sense_confusion']
    
    # Add connective statistics
    connective_freq = {
        conn: sum(counts.values())
        for conn, counts in baseline.connective_sense_counts.items()
    }
    
    output_data["task4_sense_classification"]["connective_statistics"] = {
        "most_frequent_connective": max(connective_freq.items(), key=lambda x: x[1])[0] if connective_freq else None,
        "least_frequent_connective": min(connective_freq.items(), key=lambda x: x[1])[0] if connective_freq else None,
        "avg_frequency": sum(connective_freq.values()) / len(connective_freq) if connective_freq else 0,
        "connective_frequency_distribution": dict(sorted(connective_freq.items(), key=lambda x: x[1], reverse=True))
    }
    
    # Add sense statistics
    output_data["task4_sense_classification"]["sense_statistics"] = {
        "most_common_sense": stats['global_majority_sense'],
        "sense_percentages": {
            sense: (count / sum(stats['sense_distribution'].values())) * 100
            for sense, count in stats['sense_distribution'].items()
        }
    }
    
    # Save to file
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved successfully!")
    
    return output_data


def main():
    """Main function to run Task 4 baseline."""
    print("=" * 70)
    print("Task 4: Sense Classification Baseline (Full Data)")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = "../dataset/parallel_dataset.json"
    CONNECTIVES_CSV = "../tamil_connectives.csv"
    RESULTS_OUTPUT = "task4_results.json"
    
    # Load entire dataset
    print("\n[1] Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    # Initialize baseline
    print("\n[2] Building sense mappings from entire dataset...")
    baseline = SenseClassificationBaseline()
    baseline.build_sense_mapping(dataset)
    
    # Get statistics
    print("\n[3] Dataset Statistics:")
    stats = baseline.get_statistics()
    print(f"  Number of unique connectives: {stats['num_connectives']}")
    print(f"  Number of unique senses: {stats['num_senses']}")
    print(f"  Global majority sense: {stats['global_majority_sense']}")
    print(f"  Total relations: {sum(stats['sense_distribution'].values())}")
    
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
    
    # Evaluate on the same data
    print("\n[4] Evaluating on entire dataset...")
    results = baseline.evaluate(dataset)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Print confusion matrix for top senses
    print(f"\n--- Sense Confusion Matrix (Top Senses) ---")
    confusion = results['sense_confusion']
    
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
    
    # Show sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    for i, item in enumerate(dataset[:10]):
        tamil_data = item.get("tamil", {})
        connective = tamil_data.get("connective", {}).get("raw_text", "")
        gold_sense = tamil_data.get("relation", {}).get("sense", "")
        
        if not connective:
            continue
        
        predicted_sense = baseline.predict(connective)
        is_correct = "✓" if predicted_sense == gold_sense else "✗"
        
        print(f"\nExample {i+1}: {is_correct}")
        print(f"  Connective: {connective}")
        print(f"  Gold sense: {gold_sense}")
        print(f"  Predicted:  {predicted_sense}")
        
        # Show sense distribution for this connective
        if connective in baseline.connective_sense_counts:
            sense_dist = baseline.connective_sense_counts[connective]
            print(f"  Connective sense distribution: {dict(sense_dist)}")
    
    print("\n" + "=" * 70)
    
    # Optional: Load and compare with provided CSV
    try:
        print("\n[5] Comparing with provided CSV mappings...")
        connectives_df = load_connective_sense_csv(CONNECTIVES_CSV)
        print("\nNote: This is for verification only. The baseline learns from the entire dataset.")
    except FileNotFoundError:
        print("\n[5] CSV file not found - skipping comparison")
    
    # Save results to JSON file
    print("\n[6] Saving results to JSON file...")
    output_data = save_results(results, stats, baseline, RESULTS_OUTPUT)
    
    print("\n" + "=" * 70)
    print("TASK 4 COMPLETED")
    print("=" * 70)
    print(f"\nResults summary:")
    print(f"  Overall Metrics:")
    print(f"    - Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"    - Total Samples: {results['total']}")
    print(f"    - Correct Predictions: {results['correct']}")
    print(f"    - Incorrect Predictions: {results['total'] - results['correct']}")
    
    print(f"\n  Macro-averaged Metrics:")
    print(f"    - Precision: {output_data['task4_sense_classification']['macro_metrics']['precision']:.4f}")
    print(f"    - Recall: {output_data['task4_sense_classification']['macro_metrics']['recall']:.4f}")
    print(f"    - F1-Score: {output_data['task4_sense_classification']['macro_metrics']['f1_score']:.4f}")
    
    print(f"\n  Weighted-averaged Metrics:")
    print(f"    - Precision: {output_data['task4_sense_classification']['weighted_metrics']['precision']:.4f}")
    print(f"    - Recall: {output_data['task4_sense_classification']['weighted_metrics']['recall']:.4f}")
    print(f"    - F1-Score: {output_data['task4_sense_classification']['weighted_metrics']['f1_score']:.4f}")
    
    print(f"\n  Dataset Info:")
    print(f"    - Unique Connectives: {stats['num_connectives']}")
    print(f"    - Unique Senses: {stats['num_senses']}")
    print(f"    - Most Common Sense: {stats['global_majority_sense']}")
    
    print(f"\n  Output Files:")
    print(f"    - Results JSON: {RESULTS_OUTPUT}")


if __name__ == "__main__":
    main()