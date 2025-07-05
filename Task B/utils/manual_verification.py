import cv2
import numpy as np
import matplotlib.pyplot as plt

from .embedding import get_arcface_embedding
from .preprocessing import denoise_if_needed

def verify_identity_pair(ref_img_path, test_img_path, distortion_type=None, threshold=0.5):
    # Load reference image
    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        print(f"Could not load reference image: {ref_img_path}")
        return None, None

    # Load test image
    final_input_path = denoise_if_needed(test_img_path, distortion_type)
    test_img = cv2.imread(final_input_path)
    if test_img is None:
        print(f"Could not load test image: {test_img_path}")
        return None, None

    # Get embeddings
    ref_emb = get_arcface_embedding(ref_img)
    test_emb = get_arcface_embedding(test_img, distortion_type)

    if ref_emb is None or test_emb is None:
        print("Face not detected in one of the images.")
        return None, None

    # Cosine similarity
    similarity = np.dot(ref_emb, test_emb)
    label = 1 if similarity >= threshold else 0

    # Display results
    print(f"\nCosine Similarity Score: {similarity:.4f}")
    print(f"Verification Label: {label} → {'Same Person' if label == 1 else 'Different Person'}")

    # Show side-by-side
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(ref_rgb)
    plt.title("Reference Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(test_rgb)
    plt.title("Test Image")
    plt.axis('off')

    plt.suptitle(f"{ 'Match' if label == 1 else  'No Match'} | Score: {similarity:.4f}", fontsize=14)
    plt.tight_layout()
    plt.show()

    return label, similarity
