
#  Gender Classification Using MobileNetV2

This project builds a gender classification model using a pre-trained **MobileNetV2** architecture, leveraging **TensorFlow/Keras**. The model is trained on RGB face images labeled as either **male** or **female** and evaluates performance using metrics like **accuracy**, **F1 score**, **ROC AUC**, and more. The final model can also predict gender from a new input image.



# Dataset Structure

Ensure your dataset is organized as follows:

data/
├── train/
│   ├── female/
│   │   ├── img1.jpg
│   │   └── ...
│   └── male/
│       ├── img2.jpg
│       └── ...
└── val/
    ├── female/
    └── male/

##  How to Run

### 1.  Install Required Packages

bash
pip install tensorflow opencv-python matplotlib scikit-learn


If you're using Google Colab:

python
!pip install opencv-python -q


### 2.  Load & Preprocess Dataset

Images are resized to `224x224`, converted to RGB, and preprocessed using `mobilenet_v2.preprocess_input()`.

python
X_train, y_train = load_images_for_imagenet_model(train_path)
X_val, y_val = load_images_for_imagenet_model(val_path)


### 3.  Model Architecture

 **Base**: MobileNetV2 (pre-trained on ImageNet, frozen during training)
 **Head**: GlobalAveragePooling → Dropout → Dense (Softmax for 2 classes)

python
base_model = MobileNetV2(weights='imagenet', include_top=False)

### 4. Train the Model
python
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


### 5.  Save the Trained Model

python
model.save("gender_classifier_mobilenetv2.h5")

### 6.  Evaluation Metrics

Use:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC
* ROC Curve

python
evaluate_model(model, X_val, y_val, "Validation")


### 7.  Predict Gender for a Single Image

python
predict_gender("/path/to/test/image.jpg")


It shows the image with the predicted gender and confidence.

##  Sample Output
 Validation Evaluation
 Accuracy:  0.9350
 Precision: 0.9428
 Recall:    0.9214
 F1 Score:  0.9319
 AUC-ROC:   0.9723




##  Files in the Project

| File                               | Description                                        |
| ---------------------------------- | -------------------------------------------------- |
| `train.py`                         | Contains the complete training pipeline            |
| `evaluate_model()`                 | Evaluates model performance using multiple metrics |
| `predict_gender()`                 | Predicts gender from an external image             |
| `gender_classifier_mobilenetv2.h5` | Saved trained model                                |
| `README.md`                        | You are here!                                      |


##  Requirements

* Python 3.x
* TensorFlow >= 2.0
* OpenCV
* NumPy
* Matplotlib
* scikit-learn

