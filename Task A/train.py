from config import DATASET_PATH, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH, INPUT_SHAPE, NUM_CLASSES
from model import build_model
from utils import load_data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model

# Load train/val sets
X_train, y_train = load_data(f"{DATASET_PATH}/train")
X_val, y_val = load_data(f"{DATASET_PATH}/val")

# Build and train model
model = build_model(INPUT_SHAPE, NUM_CLASSES)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Evaluate function
def evaluate_model(X, y, label="Set"):
    y_pred = model.predict(X)
    y_true = y.argmax(axis=1)
    y_pred_class = y_pred.argmax(axis=1)

    print(f"\n{label} Evaluation")
    print(classification_report(y_true, y_pred_class, target_names=["Female", "Male"]))
    auc = roc_auc_score(y_true, y_pred[:, 1])
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    plt.plot(fpr, tpr, label=f'{label} AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{label} ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.show()

# Evaluate on train and val
evaluate_model(X_train, y_train, "Train")
evaluate_model(X_val, y_val, "Validation")