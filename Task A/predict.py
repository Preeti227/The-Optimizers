import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import IMG_SIZE, MODEL_SAVE_PATH

LABEL_MAP = {0: "Female", 1: "Male"}

# Predict a single image
def predict_gender(image_path):
    # Load saved model
    try:
        model = load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found or unreadable.")
        return

    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_processed = preprocess_input(img_rgb.astype(np.float32))

    # Predict
    prediction = model.predict(np.expand_dims(img_processed, axis=0))[0]
    label = "Male" if prediction[1] > prediction[0] else "Female"
    confidence = max(prediction) * 100

    # Display
    plt.imshow(img_rgb)
    plt.title(f"{label} ({confidence:.2f}%)", color='blue')
    plt.axis('off')
    plt.show()

    print(f"Predicted: {label} with {confidence:.2f}% confidence")


# Predict all images in a folder and evaluate
def batch_predict_gender(folder_path):
    # Load model
    try:
        model = load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Gather all images
    image_paths = glob(os.path.join(folder_path, "*.jpg")) + \
                  glob(os.path.join(folder_path, "*.png")) + \
                  glob(os.path.join(folder_path, "*.jpeg"))

    if not image_paths:
        print("No images found in the specified folder.")
        return

    y_true = []
    y_pred = []


    for img_path in image_paths:
        # Get ground truth from folder name
        if "female" in folder_path.lower():
            y_true.append(0)
        elif "male" in folder_path.lower():
            y_true.append(1)
        else:
            print(f"Skipping {img_path} — can't infer label from folder name.")
            continue

        # Read and preprocess
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        img_resized = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_processed = preprocess_input(img_rgb.astype(np.float32))

        # Predict
        prediction = model.predict(np.expand_dims(img_processed, axis=0))[0]
        predicted_label = 1 if prediction[1] > prediction[0] else 0
        confidence = max(prediction) * 100
        y_pred.append(predicted_label)

        # Show prediction
        print(f"{os.path.basename(img_path)} → Predicted: {LABEL_MAP[predicted_label]} ({confidence:.2f}%)")
        plt.imshow(img_rgb)
        plt.title(f"{LABEL_MAP[predicted_label]} ({confidence:.2f}%)", color='blue')
        plt.axis('off')
        plt.show()

    # Metrics
    print("\nEvaluation Metrics:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Female", "Male"]))
