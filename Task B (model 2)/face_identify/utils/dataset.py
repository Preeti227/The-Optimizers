import os, random, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .preprocess import load_and_preprocess_image, get_identity_name

def generate_pairs(reference_dir, distorted_dir):
    X1, X2, y = [], [], []
    identity_to_ref = {}
    for fname in os.listdir(reference_dir):
        identity = os.path.splitext(fname)[0].lower()
        ref_path = os.path.join(reference_dir, fname)
        identity_to_ref[identity] = load_and_preprocess_image(ref_path)

    for dname in tqdm(os.listdir(distorted_dir)):
        if not dname.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(distorted_dir, dname)
        identity = get_identity_name(dname).lower()
        if identity not in identity_to_ref: continue

        distorted_img = load_and_preprocess_image(path)
        X1.append(identity_to_ref[identity])
        X2.append(distorted_img)
        y.append(1)

        neg_ids = [k for k in identity_to_ref if k != identity]
        for neg_id in random.sample(neg_ids, min(2, len(neg_ids))):
            X1.append(identity_to_ref[neg_id])
            X2.append(distorted_img)
            y.append(0)

    return np.array(X1), np.array(X2), np.array(y)

def show_sample_pairs(X1, X2, y, n=5):
    plt.figure(figsize=(10, 2 * n))
    for i in range(n):
        ax1 = plt.subplot(n, 2, 2 * i + 1)
        ax2 = plt.subplot(n, 2, 2 * i + 2)
        ax1.imshow(X1[i].squeeze(), cmap='gray'); ax1.set_title("Reference"); ax1.axis('off')
        ax2.imshow(X2[i].squeeze(), cmap='gray'); ax2.set_title(f"Distorted\nLabel: {y[i]}"); ax2.axis('off')
    plt.tight_layout(); plt.show()
