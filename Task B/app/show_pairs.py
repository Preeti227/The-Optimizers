import os
import sys

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.visualization import generate_pairs_from_reference, show_positive_pairs, show_negative_pairs

# Set your validation directory path
ref_dir = r"C:\Users\parth\Downloads\Task B\data\val"

# Generate positive and negative image pairs
pairs = generate_pairs_from_reference(ref_dir)

# Visualize pairs
show_positive_pairs(pairs, num_to_show=5)
show_negative_pairs(pairs, num_to_show=5)
