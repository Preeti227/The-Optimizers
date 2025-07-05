
# Gender Classification

This project builds a gender classification model using a pre-trained **MobileNetV2** architecture, leveraging **TensorFlow/Keras**. The model is trained on RGB face images labeled as either **male** or **female** and evaluates performance using metrics like **accuracy**, **F1 score**, **ROC AUC**, and more. The final model can also predict gender from a new input image.


---

## How to Run the Model

To run the gender prediction system, follow these steps:
### Step 1: Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On Mac/Linux
```

### Step 2: Install required dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Run the main interface
```bash
python main.py
```
Once main.py starts, you'll see a menu in the terminal:
Gender Prediction Project
1. Train model
2. Predict single image
3. Predict batch folder
4. Exit

---
## Model Architecture

The gender prediction model is built using a transfer learning approach with **MobileNetV2** as the feature extractor. MobileNetV2 is a lightweight convolutional neural network optimized for speed and efficiency, making it ideal for real-time and embedded applications.

The architecture is composed of the following layers:

1. **Input Layer**: Accepts RGB images of shape `(224, 224, 3)`, which are preprocessed using MobileNetV2's `preprocess_input()` function.

2. **MobileNetV2 Base**: A pretrained MobileNetV2 model is used with `include_top=False` to remove the original classification head. This base model outputs deep spatial features from the input images while keeping the number of parameters low.

3. **Global Average Pooling (GAP)**: This layer reduces the spatial dimensions of the feature map from the MobileNetV2 output by computing the average value for each feature map. It replaces dense layers to reduce overfitting and model size.

4. **Dropout Layer**: A `Dropout(rate=0.3)` is added to prevent overfitting by randomly turning off 30% of the neurons during each training step.

5. **Dense Output Layer**: A fully connected `Dense(2)` layer with `softmax` activation is used to output class probabilities for the two categories: `Male` and `Female`.

### Summary

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Head Layers**: GAP → Dropout(0.3) → Dense(2, softmax)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Input Size**: 224x224x3


###  Evaluation Metrics
* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC
* ROC Curve
