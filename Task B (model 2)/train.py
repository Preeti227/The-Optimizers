import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.dataset import generate_pairs
from models.siamese import build_siamese_network

REFERENCE_DIR = r"C:\Users\tista\OneDrive\Desktop\TASK_DATASET_MODIFIED_B\Reference"
DISTORTED_DIR = r"C:\Users\tista\OneDrive\Desktop\TASK_DATASET_MODIFIED_B\Distorted"
MODEL_PATH = "models/siamese_model.keras"

X1, X2, y = generate_pairs(REFERENCE_DIR, DISTORTED_DIR)
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

model = build_siamese_network()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X1_train, X2_train], y_train, validation_data=([X1_val, X2_val], y_val), batch_size=16, epochs=20)

if not os.path.exists("models"):
    os.makedirs("models")
model.save(MODEL_PATH)
print(f" Model saved to {MODEL_PATH}")
