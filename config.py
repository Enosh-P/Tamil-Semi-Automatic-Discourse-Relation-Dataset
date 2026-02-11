"""
Configuration file for baseline experiments
Edit these parameters to customize your experiments
"""

# Dataset paths
DATASET_PATH = "dataset/parallel_dataset.json"
CONNECTIVES_CSV = "tamil_connectives.csv"

# Dataset split parameters
TEST_SIZE = 0.2  # 20% for test, 80% for train
RANDOM_SEED = 42  # For reproducibility

# Task 1 specific settings
TASK1_RETURN_ALL_OCCURRENCES = False  # If True, detect all connectives; if False, only first

# Task 2 specific settings
TASK2_FALLBACK_STRATEGY = "global_majority"  # Options: "global_majority", "refuse"

# Evaluation settings
SHOW_SAMPLE_PREDICTIONS = 10  # Number of sample predictions to display
VERBOSE = True  # Print detailed progress

# Output settings
SAVE_PREDICTIONS = True  # Save predictions to file
TASK1_OUTPUT_FILE = "task1_predictions.json"
TASK2_OUTPUT_FILE = "task2_predictions.json"
