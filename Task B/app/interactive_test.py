import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

from utils.embedding import get_arcface_embedding
from utils.preprocessing import denoise_if_needed

from insightface.app import FaceAnalysis

DEFAULT_REF_DIR = "C:/Users/parth/Downloads/Task B/data/val"

# Initialize FaceAnalysis only once
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(160, 160))

def test_single_image_arcface():
    distorted_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter full path of distorted image: ").strip()
    threshold = 0.5

    filename = os.path.basename(distorted_path)
    distortion_type = next((d for d in ['blur', 'lowlight', 'noise', 'sunlight', 'rain', 'foggy', 'resized']
                            if d in filename.lower()), None)

    final_input_path = denoise_if_needed(distorted_path, distortion_type)
    img = cv2.imread(final_input_path)
    if img is None:
        print(f"Failed to load: {final_input_path}")
        return

    emb1 = get_arcface_embedding(img, distortion_type)
    if emb1 is None:
        print("No face detected in input image.")
        return

    best_score = -1
    best_identity = None

    for identity in sorted(os.listdir(DEFAULT_REF_DIR)):
        ref_folder = os.path.join(DEFAULT_REF_DIR, identity)
        ref_images = glob(os.path.join(ref_folder, "*.jpg"))
        for ref_path in ref_images:
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue
            faces = app.get(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            if not faces:
                continue
            emb2 = faces[0].embedding
            emb2 = emb2 / np.linalg.norm(emb2)
            similarity = np.dot(emb1, emb2)
            if similarity > best_score:
                best_score = similarity
                best_identity = identity

    label = 1 if best_score >= threshold else 0

    print(f"\nPredicted Match: {best_identity}")
    print(f"Cosine Similarity Score: {best_score:.4f}")
    print(f"Verification Label: {label} → {'Same Person' if label == 1 else 'Different Person'}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')

    if best_identity:
        ref_img_path = os.path.join(DEFAULT_REF_DIR, best_identity,
                                    os.listdir(os.path.join(DEFAULT_REF_DIR, best_identity))[0])
        ref_img = cv2.imread(ref_img_path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(ref_img)
        plt.title(f"Matched: {best_identity}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Make sure the function runs only when script is executed directly
if __name__ == "__main__":
    test_single_image_arcface()
