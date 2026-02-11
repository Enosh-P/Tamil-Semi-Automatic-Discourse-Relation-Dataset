#!/usr/bin/env python3
"""
Task 3: Multi-Connective Detection and Dataset Restructuring
Restructures dataset to group relations by sentence and evaluates detection of all connectives.
"""

import json
import pandas as pd
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split


class SentenceBasedDataset:
    """
    Restructure dataset to group multiple relations by sentence.
    Each sentence can have multiple discourse relations.
    """
    
    def __init__(self):
        """Initialize the dataset restructurer."""
        self.sentence_to_relations: Dict[str, List[Dict]] = defaultdict(list)
        self.unique_sentences: Set[str] = set()
    
    def load_from_relation_list(self, relations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Convert relation-based dataset to sentence-based dataset.
        
        Args:
            relations: List of relation objects (original format)
        
        Returns:
            Dictionary mapping sentences to list of relations
        """
        print("Restructuring dataset by sentence...")
        
        for relation in relations:
            tamil_data = relation.get("tamil", {})
            sentence = tamil_data.get("sentence", "").strip()
            
            if not sentence:
                continue
            
            # Add this relation to the sentence's list
            self.sentence_to_relations[sentence].append(tamil_data)
            self.unique_sentences.add(sentence)
        
        # Convert to regular dict for JSON serialization
        sentence_dataset = dict(self.sentence_to_relations)
        
        print(f"  Original relations: {len(relations)}")
        print(f"  Unique sentences: {len(self.unique_sentences)}")
        print(f"  Avg relations per sentence: {len(relations) / len(self.unique_sentences):.2f}")
        
        # Statistics on multi-relation sentences
        multi_relation_count = sum(1 for rels in sentence_dataset.values() if len(rels) > 1)
        print(f"  Sentences with multiple relations: {multi_relation_count}")
        
        return sentence_dataset
    
    def get_statistics(self, sentence_dataset: Dict[str, List[Dict]]) -> Dict:
        """Get detailed statistics about the sentence-based dataset."""
        stats = {
            "total_sentences": len(sentence_dataset),
            "total_relations": sum(len(rels) for rels in sentence_dataset.values()),
            "relations_per_sentence": {},
            "connective_counts": defaultdict(int),
            "sense_counts": defaultdict(int),
            "connective_type_counts": {"lexical": 0, "suffixal": 0}
        }
        
        # Count relations per sentence
        for sentence, relations in sentence_dataset.items():
            count = len(relations)
            if count not in stats["relations_per_sentence"]:
                stats["relations_per_sentence"][count] = 0
            stats["relations_per_sentence"][count] += 1
            
            # Count connectives and senses
            for relation in relations:
                tamil_data = relation.get("tamil", {})
                connective_data = tamil_data.get("connective", {})
                connective = connective_data.get("raw_text", "").strip()
                conn_type = connective_data.get("type", "lexical")
                sense = tamil_data.get("relation", {}).get("sense", "")
                
                if connective:
                    stats["connective_counts"][connective] += 1
                    stats["connective_type_counts"][conn_type] += 1
                
                if sense:
                    stats["sense_counts"][sense] += 1
        
        return stats
    
    def save_to_json(self, sentence_dataset: Dict[str, List[Dict]], output_path: str):
        """Save the sentence-based dataset to JSON."""
        print(f"\nSaving sentence-based dataset to: {output_path}")
        
        # Convert to list format for better JSON structure
        output_data = []
        for sentence, relations in sentence_dataset.items():
            output_data.append({
                "sentence": sentence,
                "relations": relations
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved {len(output_data)} sentences with {sum(len(item['relations']) for item in output_data)} relations")


class MultiConnectiveDetectionBaseline:
    """
    Baseline for detecting ALL connectives in a sentence (not just the first one).
    Evaluates how many connectives are correctly detected vs missed or false positives.
    """
    
    def __init__(self, connectives_csv: str = None):
        """
        Initialize the multi-connective detection baseline.
        
        Args:
            lexical_csv: Path to CSV containing lexical connectives
            suffixal_csv: Path to CSV containing suffixal connectives
        """
        self.lexical_connectives: Set[str] = set()
        self.suffixal_connectives: Set[str] = set()
        self.all_connectives: Set[str] = set()
        
        if connectives_csv:
            self.load_tamil_connectives(connectives_csv)
    
    def load_tamil_connectives(self, csv_path: str):
        """Load tamil connectives from CSV file."""
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
    
    def tokenize_tamil(self, sentence: str) -> List[str]:
        """Simple tokenizer for Tamil text."""
        import re
        tokens = re.findall(r'\S+', sentence)
        return tokens
    
    def detect_all_connectives(self, sentence: str) -> List[Dict]:
        """
        Detect ALL connectives in a Tamil sentence.
        
        Args:
            sentence: Tamil sentence text
        
        Returns:
            List of all detected connectives with their information
        """
        if not sentence:
            return []
        
        tokens = self.tokenize_tamil(sentence)
        detections = []
        detected_positions = set()  # Track to avoid duplicates
        
        for i, token in enumerate(tokens):
            # Check exact match (lexical connectives)
            if token in self.all_connectives:
                if i not in detected_positions:
                    connective_type = "lexical" if token in self.lexical_connectives else "suffixal"
                    
                    start_pos = sentence.find(token)
                    if start_pos != -1:
                        detection = {
                            "connective": token,
                            "type": connective_type,
                            "token_position": i,
                            "char_span_start": start_pos,
                            "char_span_end": start_pos + len(token)
                        }
                        detections.append(detection)
                        detected_positions.add(i)
            
            # Check for suffixes within tokens (suffixal connectives)
            for suffix in self.suffixal_connectives:
                if token.endswith(suffix) and suffix != token:
                    # Only add if not already detected at this position
                    if i not in detected_positions:
                        start_pos = sentence.find(token)
                        if start_pos != -1:
                            suffix_start = start_pos + len(token) - len(suffix)
                            detection = {
                                "connective": suffix,
                                "type": "suffixal",
                                "token_position": i,
                                "full_token": token,
                                "char_span_start": suffix_start,
                                "char_span_end": suffix_start + len(suffix)
                            }
                            detections.append(detection)
                            detected_positions.add(i)
                            break  # Only match one suffix per token
        
        return detections
    
    def evaluate_sentence(self, sentence: str, gold_relations: List[Dict]) -> Dict:
        """
        Evaluate connective detection for a single sentence with multiple relations.
        
        Args:
            sentence: Tamil sentence
            gold_relations: List of gold standard relations for this sentence
        
        Returns:
            Dictionary with evaluation metrics for this sentence
        """
        # Extract gold connectives
        gold_connectives = set()
        for relation in gold_relations:
            connective = relation.get("connective", {}).get("raw_text", "").strip()
            if connective:
                gold_connectives.add(connective)
        
        # Detect connectives
        predicted_detections = self.detect_all_connectives(sentence)
        predicted_connectives = set(d["connective"] for d in predicted_detections)
        
        # Calculate metrics
        true_positives = predicted_connectives & gold_connectives
        false_positives = predicted_connectives - gold_connectives
        false_negatives = gold_connectives - predicted_connectives
        
        return {
            "sentence": sentence,
            "gold_count": len(gold_connectives),
            "predicted_count": len(predicted_connectives),
            "gold_connectives": list(gold_connectives),
            "predicted_connectives": list(predicted_connectives),
            "true_positives": list(true_positives),
            "false_positives": list(false_positives),
            "false_negatives": list(false_negatives),
            "tp_count": len(true_positives),
            "fp_count": len(false_positives),
            "fn_count": len(false_negatives),
            "all_correct": len(false_positives) == 0 and len(false_negatives) == 0
        }
    
    def evaluate_dataset(self, sentence_dataset: Dict[str, List[Dict]]) -> Dict:
        """
        Evaluate the baseline on sentence-based dataset.
        
        Args:
            sentence_dataset: Dict mapping sentences to list of relations
        
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        print("\nEvaluating multi-connective detection...")
        
        total_sentences = len(sentence_dataset)
        sentence_results = []
        
        # Aggregate metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gold_connectives = 0
        total_predicted_connectives = 0
        perfect_sentences = 0
        
        for sentence, relations in sentence_dataset.items():
            result = self.evaluate_sentence(sentence, relations)
            sentence_results.append(result)
            
            total_tp += result["tp_count"]
            total_fp += result["fp_count"]
            total_fn += result["fn_count"]
            total_gold_connectives += result["gold_count"]
            total_predicted_connectives += result["predicted_count"]
            
            if result["all_correct"]:
                perfect_sentences += 1
        
        # Calculate overall metrics
        precision = total_tp / total_predicted_connectives if total_predicted_connectives > 0 else 0
        recall = total_tp / total_gold_connectives if total_gold_connectives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        sentence_accuracy = perfect_sentences / total_sentences if total_sentences > 0 else 0
        
        return {
            "total_sentences": total_sentences,
            "total_gold_connectives": total_gold_connectives,
            "total_predicted_connectives": total_predicted_connectives,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "perfect_sentences": perfect_sentences,
            "sentence_accuracy": sentence_accuracy,
            "sentence_results": sentence_results
        }


def load_dataset(json_path: str) -> List[Dict]:
    """Load the discourse relations dataset from JSON."""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} relations")
    return data


def split_sentence_dataset(sentence_dataset: Dict[str, List[Dict]], 
                           test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Split sentence-based dataset into train and test sets.
    
    Args:
        sentence_dataset: Dict mapping sentences to relations
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (train_dict, test_dict)
    """
    sentences = list(sentence_dataset.keys())
    train_sentences, test_sentences = train_test_split(
        sentences,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    train_dict = {s: sentence_dataset[s] for s in train_sentences}
    test_dict = {s: sentence_dataset[s] for s in test_sentences}
    
    train_relations = sum(len(rels) for rels in train_dict.values())
    test_relations = sum(len(rels) for rels in test_dict.values())
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dict)} sentences, {train_relations} relations")
    print(f"  Test: {len(test_dict)} sentences, {test_relations} relations")
    
    return train_dict, test_dict


def main():
    """Main function to run Task 3."""
    print("=" * 70)
    print("Task 3: Multi-Connective Detection & Dataset Restructuring")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = "../dataset/parallel_dataset.json"
    CONNECTIVES_CSV = "../tamil_connectives.csv"
    SENTENCE_DATASET_OUTPUT = "sentence_based_dataset.json"
    
    # [1] Load original relation-based dataset
    print("\n[1] Loading original dataset...")
    relations = load_dataset(DATASET_PATH)
    
    # [2] Restructure to sentence-based format
    print("\n[2] Restructuring dataset by sentence...")
    restructurer = SentenceBasedDataset()
    sentence_dataset = restructurer.load_from_relation_list(relations)
    
    # Get statistics
    print("\n[3] Dataset statistics:")
    stats = restructurer.get_statistics(sentence_dataset)
    
    print(f"  Total sentences: {stats['total_sentences']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"\n  Relations per sentence distribution:")
    for count, freq in sorted(stats['relations_per_sentence'].items()):
        percentage = (freq / stats['total_sentences']) * 100
        print(f"    {count} relation(s): {freq} sentences ({percentage:.1f}%)")
    
    print(f"\n  Connective type distribution:")
    print(f"    Lexical: {stats['connective_type_counts']['lexical']}")
    print(f"    Suffixal: {stats['connective_type_counts']['suffixal']}")
    
    print(f"\n  Top 10 most frequent connectives:")
    top_connectives = sorted(stats['connective_counts'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
    for conn, count in top_connectives:
        print(f"    {conn}: {count}")
    
    print(f"\n  Sense distribution:")
    for sense, count in sorted(stats['sense_counts'].items(), 
                               key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_relations']) * 100
        print(f"    {sense}: {count} ({percentage:.1f}%)")
    
    # [4] Save sentence-based dataset
    print("\n[4] Saving sentence-based dataset...")
    restructurer.save_to_json(sentence_dataset, SENTENCE_DATASET_OUTPUT)
    
    # [5] Split dataset
    print("\n[5] Splitting dataset (80% train, 20% test)...")
    train_dataset, test_dataset = split_sentence_dataset(
        sentence_dataset, 
        test_size=0.2, 
        random_state=42
    )
    
    # [6] Initialize multi-connective detection baseline
    print("\n[6] Initializing multi-connective detection baseline...")
    baseline = MultiConnectiveDetectionBaseline(connectives_csv=CONNECTIVES_CSV)
    
    print(f"\n  Lexicon size:")
    print(f"    Total connectives: {len(baseline.all_connectives)}")
    print(f"    Lexical: {len(baseline.lexical_connectives)}")
    print(f"    Suffixal: {len(baseline.suffixal_connectives)}")
    
    # [7] Evaluate on test set
    print("\n[7] Evaluating on test set...")
    results = baseline.evaluate_dataset(test_dataset)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nTest Set Overview:")
    print(f"  Total sentences: {results['total_sentences']}")
    print(f"  Total gold connectives: {results['total_gold_connectives']}")
    print(f"  Total predicted connectives: {results['total_predicted_connectives']}")
    
    print(f"\nDetection Performance:")
    print(f"  True Positives:  {results['total_true_positives']}")
    print(f"  False Positives: {results['total_false_positives']}")
    print(f"  False Negatives: {results['total_false_negatives']}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1 Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    
    print(f"\nSentence-Level Accuracy:")
    print(f"  Perfect sentences: {results['perfect_sentences']}/{results['total_sentences']}")
    print(f"  Accuracy: {results['sentence_accuracy']:.4f} ({results['sentence_accuracy']*100:.2f}%)")
    print(f"  (Sentences where ALL connectives were correctly detected)")
    
    # [8] Show detailed examples
    print("\n" + "=" * 70)
    print("DETAILED EXAMPLES")
    print("=" * 70)
    
    # Show examples of different outcomes
    perfect_examples = [r for r in results['sentence_results'] if r['all_correct']]
    fp_examples = [r for r in results['sentence_results'] if r['fp_count'] > 0]
    fn_examples = [r for r in results['sentence_results'] if r['fn_count'] > 0]
    multi_conn_examples = [r for r in results['sentence_results'] if r['gold_count'] > 1]
    
    print("\n--- Perfect Detections (First 3) ---")
    for i, result in enumerate(perfect_examples[:3], 1):
        print(f"\nExample {i}:")
        print(f"  Sentence: {result['sentence'][:80]}...")
        print(f"  Gold: {result['gold_connectives']}")
        print(f"  Predicted: {result['predicted_connectives']}")
        print(f"  ✓ All correct!")
    
    print("\n--- False Positives (First 3) ---")
    for i, result in enumerate(fp_examples[:3], 1):
        print(f"\nExample {i}:")
        print(f"  Sentence: {result['sentence'][:80]}...")
        print(f"  Gold: {result['gold_connectives']}")
        print(f"  Predicted: {result['predicted_connectives']}")
        print(f"  False Positives: {result['false_positives']}")
    
    print("\n--- False Negatives (First 3) ---")
    for i, result in enumerate(fn_examples[:3], 1):
        print(f"  \nExample {i}:")
        print(f"  Sentence: {result['sentence'][:80]}...")
        print(f"  Gold: {result['gold_connectives']}")
        print(f"  Predicted: {result['predicted_connectives']}")
        print(f"  Missed: {result['false_negatives']}")
    
    print("\n--- Multi-Connective Sentences (First 3) ---")
    for i, result in enumerate(multi_conn_examples[:3], 1):
        print(f"\nExample {i}:")
        print(f"  Sentence: {result['sentence'][:80]}...")
        print(f"  Number of connectives: {result['gold_count']}")
        print(f"  Gold: {result['gold_connectives']}")
        print(f"  Predicted: {result['predicted_connectives']}")
        print(f"  TP: {result['tp_count']}, FP: {result['fp_count']}, FN: {result['fn_count']}")
    
    # [9] Save detailed results
    print("\n" + "=" * 70)
    output_results = {
        "configuration": {
            "test_size": 0.2,
            "lexicon_size": len(baseline.all_connectives)
        },
        "overview": {
            "total_sentences": results['total_sentences'],
            "total_gold_connectives": results['total_gold_connectives'],
            "total_predicted_connectives": results['total_predicted_connectives'],
        },
        "metrics": {
            "precision": results['precision'],
            "recall": results['recall'],
            "f1_score": results['f1_score'],
            "sentence_accuracy": results['sentence_accuracy']
        },
        "confusion": {
            "true_positives": results['total_true_positives'],
            "false_positives": results['total_false_positives'],
            "false_negatives": results['total_false_negatives']
        }
    }
    
    results_output_path = "task3_results.json"
    with open(results_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
