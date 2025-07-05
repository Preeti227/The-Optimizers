import cv2
import numpy as np
import os

IMAGE_SIZE = (160, 160)

def load_and_preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def get_identity_name(filename):
    parts = filename.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else os.path.splitext(filename)[0]