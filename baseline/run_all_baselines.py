#!/usr/bin/env python3
"""
Run both baseline tasks sequentially
"""

import json
import sys
from typing import Dict, List
from task1_connective_detection_baseline import (
    ConnectiveDetectionBaseline,
    load_dataset,
    split_dataset
)
from task2_sense_classification_baseline import SenseClassificationBaseline
from task3_multi_connective_detection import (
    SentenceBasedDataset,
    MultiConnectiveDetectionBaseline,
    split_sentence_dataset
)

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def run_all_three_tasks(dataset_path: str, connectives_csv: str, 
                   test_size: float = 0.2, random_state: int = 42):
    """
    Run both Task 1 and Task 2 baselines sequentially.
    
    Args:
        dataset_path: Path to JSON dataset
        connectives_csv: Path to tamil connectives CSV
        test_size: Test set proportion
        random_state: Random seed
    """
    print("=" * 80)
    print("RUNNING BOTH BASELINE TASKS")
    print("=" * 80)
    
    # Load and split dataset
    print("\n[SETUP] Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    print("\n[SETUP] Splitting dataset...")
    train_data, test_data = split_dataset(dataset, test_size=test_size, random_state=random_state)
    
    # ========== TASK 1: CONNECTIVE DETECTION ==========
    print("\n" + "=" * 80)
    print("TASK 1: CONNECTIVE DETECTION")
    print("=" * 80)
    
    print("\n[Task 1] Initializing baseline...")
    task1_baseline = ConnectiveDetectionBaseline(connectives_csv=connectives_csv)
    
    print(f"\n[Task 1] Lexicon statistics:")
    print(f"  Total connectives: {len(task1_baseline.all_connectives)}")
    print(f"  Lexical: {len(task1_baseline.lexical_connectives)}")
    print(f"  Suffixal: {len(task1_baseline.suffixal_connectives)}")
    
    print("\n[Task 1] Evaluating...")
    task1_results = task1_baseline.evaluate(test_data)
    
    print("\n[Task 1] Results:")
    print(f"  Accuracy:  {task1_results['accuracy']:.4f} ({task1_results['accuracy']*100:.2f}%)")
    print(f"  Precision: {task1_results['precision']:.4f} ({task1_results['precision']*100:.2f}%)")
    print(f"  Recall:    {task1_results['recall']:.4f} ({task1_results['recall']*100:.2f}%)")
    print(f"  F1 Score:  {task1_results['f1_score']:.4f} ({task1_results['f1_score']*100:.2f}%)")
    print(f"  Exact Match: {task1_results['exact_match_accuracy']:.4f} ({task1_results['exact_match_accuracy']*100:.2f}%)")
    
    # ========== TASK 2: SENSE CLASSIFICATION ==========
    print_section_header("TASK 2: SENSE CLASSIFICATION")
    
    print("\n[Task 2] Training baseline...")
    task2_baseline = SenseClassificationBaseline()
    task2_baseline.train(train_data)
    
    stats = task2_baseline.get_statistics()
    print(f"\n[Task 2] Training statistics:")
    print(f"  Unique connectives: {stats['num_connectives']}")
    print(f"  Unique senses: {stats['num_senses']}")
    print(f"  Global majority sense: {stats['global_majority_sense']}")
    
    # Show top sense distribution
    print(f"\n  Top 5 senses:")
    sense_dist = sorted(stats['sense_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
    for sense, count in sense_dist:
        percentage = (count / sum(stats['sense_distribution'].values())) * 100
        print(f"    {sense}: {count} ({percentage:.1f}%)")
    
    print("\n[Task 2] Evaluating...")
    task2_results = task2_baseline.evaluate(test_data)
    
    print("\n[Task 2] Results:")
    print(f"  Overall Accuracy: {task2_results['accuracy']:.4f} ({task2_results['accuracy']*100:.2f}%)")
    print(f"  Seen Connectives: {task2_results['seen_accuracy']:.4f} ({task2_results['seen_accuracy']*100:.2f}%) [n={task2_results['seen_total']}]")
    print(f"  Unseen Connectives: {task2_results['unseen_accuracy']:.4f} ({task2_results['unseen_accuracy']*100:.2f}%) [n={task2_results['unseen_total']}]")
    
    # ========== TASK 3: MULTI-CONNECTIVE DETECTION ==========
    print_section_header("TASK 3: MULTI-CONNECTIVE DETECTION")
    
    print("\n[Task 3] Restructuring dataset by sentence...")
    restructurer = SentenceBasedDataset()
    sentence_dataset = restructurer.load_from_relation_list(dataset)
    
    # Get statistics
    dataset_stats = restructurer.get_statistics(sentence_dataset)
    print(f"\n[Task 3] Dataset statistics:")
    print(f"  Total sentences: {dataset_stats['total_sentences']}")
    print(f"  Total relations: {dataset_stats['total_relations']}")
    print(f"  Avg relations/sentence: {dataset_stats['total_relations'] / dataset_stats['total_sentences']:.2f}")
    
    # Show distribution
    print(f"\n  Relations per sentence distribution:")
    for count in sorted(dataset_stats['relations_per_sentence'].keys())[:5]:
        freq = dataset_stats['relations_per_sentence'][count]
        percentage = (freq / dataset_stats['total_sentences']) * 100
        print(f"    {count} relation(s): {freq} sentences ({percentage:.1f}%)")
    
    # Split sentence-based dataset
    print("\n[Task 3] Splitting sentence-based dataset...")
    train_sentences, test_sentences = split_sentence_dataset(
        sentence_dataset, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Initialize multi-connective baseline
    print("\n[Task 3] Initializing multi-connective detection baseline...")
    task3_baseline = MultiConnectiveDetectionBaseline(connectives_csv=connectives_csv)
    
    # Evaluate
    print("\n[Task 3] Evaluating...")
    task3_results = task3_baseline.evaluate_dataset(test_sentences)
    
    print("\n[Task 3] Results:")
    print(f"  Precision: {task3_results['precision']:.4f} ({task3_results['precision']*100:.2f}%)")
    print(f"  Recall:    {task3_results['recall']:.4f} ({task3_results['recall']*100:.2f}%)")
    print(f"  F1 Score:  {task3_results['f1_score']:.4f} ({task3_results['f1_score']*100:.2f}%)")
    print(f"  Sentence Accuracy: {task3_results['sentence_accuracy']:.4f} ({task3_results['sentence_accuracy']*100:.2f}%)")
    print(f"\n  Connectives detected: {task3_results['total_predicted_connectives']}")
    print(f"  True Positives:  {task3_results['total_true_positives']}")
    print(f"  False Positives: {task3_results['total_false_positives']}")
    print(f"  False Negatives: {task3_results['total_false_negatives']}")
    
    # ========== CROSS-TASK ANALYSIS ==========
    print_section_header("CROSS-TASK ANALYSIS")
    
    print("\n1. Task 1 vs Task 3 Comparison:")
    print("   (First-occurrence detection vs All-connectives detection)")
    print(f"   Task 1 F1: {task1_results['f1_score']:.4f}")
    print(f"   Task 3 F1: {task3_results['f1_score']:.4f}")
    print(f"   Difference: {abs(task1_results['f1_score'] - task3_results['f1_score']):.4f}")
    print(f"   → Task 3 is {'harder' if task3_results['f1_score'] < task1_results['f1_score'] else 'easier'} (must find ALL connectives)")
    
    print("\n2. End-to-End Pipeline (Task 1 → Task 2):")
    print("   (Detect connective, then classify its sense)")
    
    correct_detection_and_sense = 0
    total_with_connective = 0
    
    for item in test_data:
        tamil_data = item.get("tamil", {})
        sentence = tamil_data.get("sentence", "")
        gold_connective = tamil_data.get("connective", {}).get("raw_text", "")
        gold_sense = tamil_data.get("relation", {}).get("sense", "")
        
        if not gold_connective:
            continue
        
        total_with_connective += 1
        
        # Task 1: Detect connective
        task1_pred = task1_baseline.predict(sentence)
        
        if task1_pred["has_connective"] and task1_pred["detections"]:
            detected_conn = task1_pred["detections"][0]["connective"]
            
            # Check if correctly detected
            if detected_conn == gold_connective:
                # Task 2: Classify sense
                predicted_sense = task2_baseline.predict(detected_conn)
                
                # Check if sense is correct
                if predicted_sense == gold_sense:
                    correct_detection_and_sense += 1
    
    end_to_end_accuracy = correct_detection_and_sense / total_with_connective if total_with_connective > 0 else 0
    
    print(f"   Correct detection AND sense: {correct_detection_and_sense}/{total_with_connective}")
    print(f"   End-to-end accuracy: {end_to_end_accuracy:.4f} ({end_to_end_accuracy*100:.2f}%)")
    
    print("\n3. Multi-Relation Sentence Analysis:")
    multi_rel_count = sum(1 for rels in test_sentences.values() if len(rels) > 1)
    multi_rel_total = len([r for r in task3_results['sentence_results'] if r['gold_count'] > 1])
    
    if multi_rel_total > 0:
        multi_rel_perfect = len([r for r in task3_results['sentence_results'] 
                                 if r['gold_count'] > 1 and r['all_correct']])
        multi_rel_accuracy = multi_rel_perfect / multi_rel_total
        
        print(f"   Sentences with multiple connectives: {multi_rel_count}")
        print(f"   Perfect detection rate: {multi_rel_perfect}/{multi_rel_total} ({multi_rel_accuracy*100:.2f}%)")
        print(f"   → Multi-connective sentences are {'harder' if multi_rel_accuracy < task3_results['sentence_accuracy'] else 'as easy'}")
    
    # ========== COMPREHENSIVE SUMMARY ==========
    print_section_header("COMPREHENSIVE SUMMARY")
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train/Test split: {len(train_data)}/{len(test_data)} ({test_size*100:.0f}% test)")
    print(f"  Unique sentences: {dataset_stats['total_sentences']}")
    print(f"  Sentences with multiple relations: {sum(1 for c, f in dataset_stats['relations_per_sentence'].items() if c > 1 for _ in range(f))}")
    
    print(f"\nLexicon Coverage:")
    print(f"  Total connectives in lexicon: {len(task1_baseline.all_connectives)}")
    print(f"  Lexical: {len(task1_baseline.lexical_connectives)}")
    print(f"  Suffixal: {len(task1_baseline.suffixal_connectives)}")
    
    print(f"\n┌─────────────────────────────────────────────────────────────────┐")
    print(f"│                      PERFORMANCE SUMMARY                        │")
    print(f"├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Task 1 (First Connective Detection)                            │")
    print(f"│   F1 Score:  {task1_results['f1_score']:6.4f} ({task1_results['f1_score']*100:5.2f}%)                                │")
    print(f"│   Accuracy:  {task1_results['accuracy']:6.4f} ({task1_results['accuracy']*100:5.2f}%)                                │")
    print(f"│                                                                 │")
    print(f"│ Task 2 (Sense Classification)                                  │")
    print(f"│   Overall:   {task2_results['accuracy']:6.4f} ({task2_results['accuracy']*100:5.2f}%)                                │")
    print(f"│   Seen:      {task2_results['seen_accuracy']:6.4f} ({task2_results['seen_accuracy']*100:5.2f}%)                                │")
    print(f"│   Unseen:    {task2_results['unseen_accuracy']:6.4f} ({task2_results['unseen_accuracy']*100:5.2f}%)                                │")
    print(f"│                                                                 │")
    print(f"│ Task 3 (Multi-Connective Detection)                            │")
    print(f"│   F1 Score:  {task3_results['f1_score']:6.4f} ({task3_results['f1_score']*100:5.2f}%)                                │")
    print(f"│   Sent Acc:  {task3_results['sentence_accuracy']:6.4f} ({task3_results['sentence_accuracy']*100:5.2f}%)                                │")
    print(f"│                                                                 │")
    print(f"│ End-to-End Pipeline (Task 1 → Task 2)                          │")
    print(f"│   Combined:  {end_to_end_accuracy:6.4f} ({end_to_end_accuracy*100:5.2f}%)                                │")
    print(f"└─────────────────────────────────────────────────────────────────┘")
    
    # ========== SAVE COMPREHENSIVE RESULTS ==========
    summary = {
        "dataset": {
            "total_samples": len(dataset),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "unique_sentences": dataset_stats['total_sentences'],
            "test_split_ratio": test_size,
            "random_seed": random_state
        },
        "lexicon": {
            "total": len(task1_baseline.all_connectives),
            "lexical": len(task1_baseline.lexical_connectives),
            "suffixal": len(task1_baseline.suffixal_connectives)
        },
        "task1_first_connective": {
            "accuracy": task1_results['accuracy'],
            "precision": task1_results['precision'],
            "recall": task1_results['recall'],
            "f1_score": task1_results['f1_score'],
            "exact_match_accuracy": task1_results['exact_match_accuracy']
        },
        "task2_sense_classification": {
            "overall_accuracy": task2_results['accuracy'],
            "seen_accuracy": task2_results['seen_accuracy'],
            "unseen_accuracy": task2_results['unseen_accuracy'],
            "seen_count": task2_results['seen_total'],
            "unseen_count": task2_results['unseen_total'],
            "num_connectives_in_training": stats['num_connectives'],
            "num_senses": stats['num_senses']
        },
        "task3_multi_connective": {
            "precision": task3_results['precision'],
            "recall": task3_results['recall'],
            "f1_score": task3_results['f1_score'],
            "sentence_accuracy": task3_results['sentence_accuracy'],
            "total_sentences": task3_results['total_sentences'],
            "perfect_sentences": task3_results['perfect_sentences'],
            "total_true_positives": task3_results['total_true_positives'],
            "total_false_positives": task3_results['total_false_positives'],
            "total_false_negatives": task3_results['total_false_negatives']
        },
        "end_to_end_pipeline": {
            "accuracy": end_to_end_accuracy,
            "correct": correct_detection_and_sense,
            "total": total_with_connective
        }
    }
    
    output_file = "all_tasks_comprehensive_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Comprehensive results saved to: {output_file}")
    print("=" * 80)
    
    return summary


def main():
    """Main entry point."""
    # Default configuration
    DATASET_PATH = "../dataset/parallel_dataset.json"
    CONNECTIVES_CSV = "../tamil_connectives.csv"
    
    # Allow command line arguments
    if len(sys.argv) >= 2:
        DATASET_PATH = sys.argv[1]
    if len(sys.argv) >= 3:
        CONNECTIVES_CSV = sys.argv[2]
    
    try:
        run_all_three_tasks(
            dataset_path=DATASET_PATH,
            connectives_csv=CONNECTIVES_CSV,
            test_size=0.2,
            random_state=42
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nUsage:")
        print("  python run_all_baselines.py [dataset.json] [lexical.csv] [suffixal.csv]")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
