import os
import sys
import pandas as pd
# Add parent directory to path so utils works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.matching import batch_arcface_identity_match

# Define your test and reference directories
test_dir = r"C:\Users\parth\Downloads\Task B\data\test"
ref_dir = r"C:\Users\parth\Downloads\Task B\data\val"

# Run batch evaluation
results = batch_arcface_identity_match(test_dir, ref_dir)

# Print results
print("\nBatch Evaluation Summary")
print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
print(f"Macro F1-score: {results['f1_macro']:.4f}")
print(f"Total Samples: {len(results['ground_truth'])}")

#  Saving results to CSV

df = pd.DataFrame({
    "Ground Truth": results['ground_truth'],
    "Prediction": results['predictions'],
    "Score": results['scores']
})
df.to_csv("batch_results.csv", index=False)
print("Results saved to batch_results.csv")