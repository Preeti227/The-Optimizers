import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.preprocess import load_and_preprocess_image
from evaluation.metrices import evaluate_model
from PIL import Image

MODEL_PATH = "models/siamese_model.keras"

# Function used in model
def abs_diff(tensors):
    x, y = tensors
    return tf.abs(x - y)

def show_images(img1_path, img2_path, prediction_text):
    """Display the two input images side by side with prediction label."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1 (Reference)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2 (Distorted)")
    plt.axis("off")

    # Show prediction text
    plt.suptitle(f"Prediction: {prediction_text}", fontsize=16, color="green" if "SAME" in prediction_text else "red")
    plt.tight_layout()
    plt.show()

def verify_pair(img1_path, img2_path, model, threshold=0.4):
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    score = model.predict([img1, img2])[0][0]
    print(f"\n Confidence Score: {score:.4f}")
    
    if score > threshold:
        result_text = " SAME person"
    else:
        result_text = " DIFFERENT persons"

    print(result_text)

    return np.array([img1.squeeze()]), np.array([img2.squeeze()]), np.array([1 if score > threshold else 0]), result_text

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train first.")
    
    model = load_model(MODEL_PATH, custom_objects={'abs_diff': abs_diff})
    
    img1_path = input("Enter path of the Reference or clear image: ").strip()
    img2_path = input("Enter path of the Distorted image: ").strip()

    # 🔧 Use threshold = 0.4
    X1, X2, y_true, result_text = verify_pair(img1_path, img2_path, model, threshold=0.4)

    show_images(img1_path, img2_path, result_text)

    evaluate_model(model, X1, X2, y_true, threshold=0.4)
