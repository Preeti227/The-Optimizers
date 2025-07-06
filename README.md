**TEAM NAME: THE OPTIMIZERS**

Here’s a **combined `README.md`** that documents **two different approaches** for **Task B: Face Matching**:

1. **Approach 1**: Siamese Neural Network with Ensemble
2. **Approach 2**: ArcFace (InsightFace) Embedding-based Matching

This README clearly separates both pipelines and describes their usage, scripts, and architecture under a single project.


# Task B: Robust Face Matching Under Distortions

This repository presents **two different approaches** for **face verification under adverse visual conditions**, as part of **Task B**. Each approach uses a different strategy for comparing distorted facial images with their clean counterparts.


##  Approach 1: Siamese Neural Network with Ensemble

This method uses a deep Siamese architecture with parallel embedding streams and a final classifier.


### Dataset Summary

- 50 unique identities
- Each identity has:
  - 1 clear image (in `reference/`)
  - 5 distorted images (in `distorted/`)
- Distortions include blur, fog, resizing, noise, and rain
- Images are resized to `160×160`, converted to grayscale, denoised, and normalized

###  Model Details

- Siamese network with **three embedding paths**:
  - CNN branch
  - Frozen ResNet50
  - FaceNet-like shallow network
- Embeddings are concatenated and compared via **absolute difference**
- Final sigmoid classifier predicts similarity
- Trained with **binary cross-entropy** and **Adam optimizer**

### Performance

- **Top-1 Accuracy**: `0.852`
- **Macro Precision**: `0.89`
- **Macro Recall**: `0.93`
- **F1 Score**: `0.90`

###  How to Run

bash
python run.py       # Run on a single image pair
python train.py     # Train the Siamese model

##  Approach 2: ArcFace Embedding-Based Matching

This approach uses **pretrained ArcFace models** from InsightFace to extract embeddings and compares them using cosine similarity.

###  Key Components

* Embeddings extracted using **ArcFace (ResNet50)** via `insightface`
* Inputs are optionally denoised using Gaussian/median filtering
* Final similarity computed using **cosine distance**
* Threshold (default = 0.5) used for verification

###  Scripts & Usage

| Task                         | Script                    | Description                                            |
| ---------------------------- | ------------------------- | ------------------------------------------------------ |
| Manual Identity Verification | `app/manual_test.py`      | Match one distorted image with one reference           |
| Batch Evaluation             | `app/batch_eval.py`       | Compare all distorted images with reference identities |
| Interactive Single Test      | `app/interactive_test.py` | Test one image against full reference set (val)        |
| Visualize Pairs              | `app/show_pairs.py`       | Show positive/negative image pairs                     |

###  Evaluation Metrics

* **Top-1 Accuracy**
* **Macro F1 Score**

###  Requirements

```bash
pip install -r requirements.txt
```

## Summary

| Feature             | Siamese Ensemble                      | ArcFace Embedding Matching    |
| ------------------- | ------------------------------------- | ----------------------------- |
| Model Type          | Siamese Neural Network + Ensemble     | Pretrained ArcFace (ResNet50) |
| Input Preprocessing | Resize, grayscale, filters, normalize | Resize, filters, normalize    |
| Similarity Measure  | Absolute diff + sigmoid classifier    | Cosine similarity             |
| Training Required   | Yes                                 | No (pretrained)             |
| Evaluation Support  | Yes                                 | Yes                         |
| Visual Output       | Matplotlib (image pair display)       | Matplotlib (pair viewer)      |


## A seperate `Readme.md` is provided for both the aproaches

