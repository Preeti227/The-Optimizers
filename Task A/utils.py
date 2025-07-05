import os
import cv2
import numpy as np
from glob import glob
from config import IMG_SIZE
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

def load_data(folder_path):
    images, labels = [], []
    for label_name, label_id in [('female', 0), ('male', 1)]:
        folder = os.path.join(folder_path, label_name)
        for fname in glob(os.path.join(folder, "*")):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img.astype(np.float32))
            images.append(img)
            labels.append(label_id)

    return np.array(images), to_categorical(labels)