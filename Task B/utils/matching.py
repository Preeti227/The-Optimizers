import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, f1_score
from .embedding import get_arcface_embedding
from .preprocessing import denoise_if_needed

def batch_arcface_identity_match(test_root_dir, ref_dir, threshold=0.5):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(160, 160))

    all_preds, all_labels, all_scores = [], [], []
    print("Starting batch evaluation...")

    for identity in tqdm(sorted(os.listdir(test_root_dir))):
        identity_path = os.path.join(test_root_dir, identity)
        if not os.path.isdir(identity_path):
            continue

        for distortion in os.listdir(identity_path):
            distortion_folder = os.path.join(identity_path, distortion)
            if not os.path.isdir(distortion_folder):
                continue

            for img_name in os.listdir(distortion_folder):
                img_path = os.path.join(distortion_folder, img_name)
                distortion_type = distortion.lower()
                final_input_path = denoise_if_needed(img_path, distortion_type)
                img = cv2.imread(final_input_path)
                if img is None:
                    continue

                emb1 = get_arcface_embedding(img, distortion_type)
                if emb1 is None:
                    continue

                best_score = -1
                best_identity = None

                for ref_id in sorted(os.listdir(ref_dir)):
                    ref_folder = os.path.join(ref_dir, ref_id)
                    ref_imgs = glob(os.path.join(ref_folder, "*.jpg"))
                    for ref_img_path in ref_imgs:
                        ref_img = cv2.imread(ref_img_path)
                        if ref_img is None:
                            continue
                        ref_faces = app.get(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
                        if not ref_faces:
                            continue
                        emb2 = ref_faces[0].embedding
                        emb2 = emb2 / np.linalg.norm(emb2)
                        similarity = np.dot(emb1, emb2)
                        if similarity > best_score:
                            best_score = similarity
                            best_identity = ref_id

                all_labels.append(identity)
                all_preds.append(best_identity)
                all_scores.append(best_score)

                print(f"{img_name} | GT: {identity} | Pred: {best_identity} | Score: {best_score:.4f}")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\nTop-1 Accuracy: {acc:.4f}")
    print(f"Macro-averaged F1-score: {f1:.4f}")
    print(f"Total Samples Evaluated: {len(all_labels)}")

    return {
        'predictions': all_preds,
        'ground_truth': all_labels,
        'scores': all_scores,
        'top1_accuracy': acc,
        'f1_macro': f1
    }
