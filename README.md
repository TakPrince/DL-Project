# Kirana Store Shelf Product Recognition Using CNN

**Student:** Prince Kumar Tak | **ID:** ADT25SOCM0132 | **Course Unit:** Unit III | **Competencies:** CO2, CO4

A small, lightweight, and explainable Convolutional Neural Network (CNN) engineered to recognize packaged grocery items (`Juice`, `Milk`, `Yoghurt`) from store shelves. The project implements a complete, leak-free deep learning pipeline: data preparation, stratified splitting, in-model preprocessing, custom CNN architecture, deterministic training, rigorous offline evaluation, a command-line inference tool, and an interactive local live model web demonstration.

---

## 1. Problem & Project Objective

Kirana stores and retail supermarkets handle hundreds of packaged goods daily. Inventory monitoring, shelf auditing, and billing automation require accurate visual recognition of packaged products despite variations in lighting, rotation, and slight angle distortions.

### Project Scope
- **Core Deliverable:** End-to-end 3-class CNN image classification (`Juice`, `Milk`, `Yoghurt`).
- **Pipeline:** `Raw Data Inspection → Stratified Splitting → In-Model Preprocessing → Custom CNN → Training & Early Stopping → Offline Evaluation → CLI Inference → Live Model Demo`.
- **Design Philosophy:** Minimalist, viva-friendly, zero external heavy dependencies, strictly reproducible with fixed seeds.

---

## 2. Dataset & Stratified Splits

### Dataset Source
- **Origin:** Marcus Klasson et al., *Grocery Store Dataset* (WACV 2019).
- **Format:** High-quality real-world packaged product photographs converted to 3-channel standard RGB JPEG.

### Class Counts & Split Ratios
A balanced subset of **120 images per class** (360 total images) was constructed and partitioned physically on disk before any modeling:

| Class | Total Images | Train Set (70%) | Validation Set (15%) | Test Set (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Juice** | 120 | 84 | 18 | 18 |
| **Milk** | 120 | 84 | 18 | 18 |
| **Yoghurt** | 120 | 84 | 18 | 18 |
| **Total** | **360** | **252** | **54** | **54** |

### Leakage Prevention & Integrity Verification
1. **Physical Separation:** Files are segregated into `data/train/`, `data/val/`, and `data/test/` folders.
2. **Zero Overlap:** Image SHA-256 hashes and file paths were verified to ensure zero overlap across training, validation, and testing sets.
3. **Strict Hold-Out Test Set:** The test set was never used during training, hyperparameter tuning, or early stopping decisions; it was loaded exclusively for final model evaluation.
4. **Sample Size Note:** With 18 images per class in the test set (54 total), test metrics are indicative and informative for evaluating core classification capacity.

---

## 3. Preprocessing & Data Augmentation

Data pipelines are configured in `src/data_pipeline.py` using `tf.keras.utils.image_dataset_from_directory`:
- **Input Resolution:** Resized to $128 \times 128 \times 3$ RGB.
- **Batch Size:** 32.
- **Label Encoding:** Integer encoding mapped via `models/class_names.json`.
- **Deterministic Shuffling:**
  - Training dataset: `shuffle=True`, `seed=42`.
  - Validation dataset: `shuffle=False` (prevents label misalignment).
  - Test dataset: `shuffle=False` (guarantees accurate confusion matrices).
- **In-Model Normalization & Augmentation:**
  - Integrated inside `src/model.py` so external callers need not perform manual normalization.
  - Normalization: `layers.Rescaling(1.0 / 255.0)` transforms pixel values from $[0, 255]$ to $[0.0, 1.0]$.
  - Augmentation (training only): `RandomRotation(0.05)`, `RandomZoom(0.05)`, and `RandomFlip("horizontal")`. Kept mild to preserve product logo and label legibility.

---

## 4. CNN Architecture

The architecture is implemented in `src/model.py` as an explainable, 3-block convolutional network:

```
Input (128 × 128 × 3 RGB)
  │
  ├── [Augmentation] RandomRotation(±5%) + RandomZoom(±5%) + RandomFlip(horizontal)
  ├── [Normalization] Rescaling(1/255.0)
  │
  ├── [Conv Block 1] Conv2D(32 filters, 3×3 kernel, ReLU) ──> MaxPooling2D(2×2)
  ├── [Conv Block 2] Conv2D(64 filters, 3×3 kernel, ReLU) ──> MaxPooling2D(2×2)
  ├── [Conv Block 3] Conv2D(128 filters, 3×3 kernel, ReLU) ──> MaxPooling2D(2×2)
  │
  ├── [Flatten] Reshapes 14×14×128 feature map to 25,088 vector
  ├── [Dense] 128 units, ReLU activation
  ├── [Dropout] Rate = 0.5 (regularization)
  └── [Output Head] Dense(3 units, Softmax activation)
```

### Model Parameter Breakdown
```
Model: "baseline_grocery_cnn"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ aug_random_rotation             │ (None, 128, 128, 3)    │             0 │
│ aug_random_zoom                 │ (None, 128, 128, 3)    │             0 │
│ aug_random_flip                 │ (None, 128, 128, 3)    │             0 │
│ rescaling_norm (Rescaling)      │ (None, 128, 128, 3)    │             0 │
│ conv2d_block1 (Conv2D)          │ (None, 126, 126, 32)   │           896 │
│ maxpool_block1 (MaxPooling2D)   │ (None, 63, 63, 32)     │             0 │
│ conv2d_block2 (Conv2D)          │ (None, 61, 61, 64)     │        18,496 │
│ maxpool_block2 (MaxPooling2D)   │ (None, 30, 30, 64)     │             0 │
│ conv2d_block3 (Conv2D)          │ (None, 28, 28, 128)    │        73,856 │
│ maxpool_block3 (MaxPooling2D)   │ (None, 14, 14, 128)    │             0 │
│ flatten (Flatten)               │ (None, 25088)          │             0 │
│ dense_dense128 (Dense)          │ (None, 128)            │     3,211,392 │
│ dropout_regularization (Dropout)│ (None, 128)            │             0 │
│ output_probabilities (Dense)    │ (None, 3)              │           387 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,305,027 (12.61 MB)
 Trainable params: 3,305,027 (12.61 MB)
 Non-trainable params: 0 (0.00 B)
```

---

## 5. Training Configuration & Dynamics

- **Optimizer:** Adam ($\beta_1=0.9$, $\beta_2=0.999$, learning rate $= 0.001$).
- **Loss Function:** Sparse Categorical Crossentropy.
- **Batch Size:** 32.
- **Max Epochs:** 20.
- **Early Stopping:** Monitored `val_loss`, patience of 4 epochs, automatically restored best weights.
- **Checkpointing:** Best checkpoint saved to `models/best_model.keras`.
- **Training Time:** ~33 seconds on CPU.
- **Convergence Details:** Training stopped at **Epoch 14**. The minimum validation loss was achieved at **Epoch 10** (`val_loss = 0.1659`, `val_accuracy = 96.30%`).

---

## 6. Real Model Performance & Metrics

All metrics reported here were calculated by `src/evaluate.py` by evaluating `models/best_model.keras` on the untouched test split:

### Summary Performance Table
| Metric | Real Model Value | Chance Baseline |
| :--- | :---: | :---: |
| **Baseline Accuracy (Random Chance)** | 33.33% | $1 / 3$ classes |
| **Best Training Accuracy (Epoch 10)** | 88.89% | — |
| **Best Validation Accuracy (Epoch 10)**| **96.30%** | — |
| **Best Validation Loss (Epoch 10)**   | **0.1659** | — |
| **Final Test Accuracy**               | **94.44%** (51 / 54 correct) | > 70% |
| **Final Test Loss**                   | **0.1867** | — |

### Detailed Classification Report (Test Set)
```
              precision    recall  f1-score   support

       Juice     0.8571    1.0000    0.9231        18
        Milk     1.0000    0.9444    0.9714        18
     Yoghurt     1.0000    0.8889    0.9412        18

    accuracy                         0.9444        54
   macro avg     0.9524    0.9444    0.9452        54
weighted avg     0.9524    0.9444    0.9452        54
```

### Confusion Matrix
```
Rows: True Class | Columns: Predicted Class ['Juice', 'Milk', 'Yoghurt']

                 Juice    Milk    Yoghurt
True Juice         18       0        0
True Milk           1      17        0
True Yoghurt        2       0       16
```

### Error & Overfitting Analysis
1. **Juice (100% Recall, 85.7% Precision):** Every single Juice test sample was correctly identified. However, 1 Milk carton and 2 Yoghurt drinks were misclassified as Juice. This occurs because cylindrical drinkable yoghurt bottles and rectangular milk cartons share packaging silhouettes and color schemes common to fruit juice containers.
2. **Milk & Yoghurt (100% Precision):** Zero false positives occurred for Milk and Yoghurt; when the model predicted Milk or Yoghurt, it was 100% correct.
3. **Overfitting Assessment:** Training accuracy (~89–91%) and validation accuracy (~96%) tracked closely. The presence of `Dropout(0.5)` and in-model data augmentation prevented large divergence between training and validation loss curves.

---

## 7. Project Structure

```
DL-Project/
├── data/
│   ├── raw/                      # Downloaded source images per class
│   ├── train/                    # 252 training images (84/class)
│   ├── val/                      # 54 validation images (18/class)
│   └── test/                     # 54 test images (18/class)
├── models/
│   ├── best_model.keras          # Saved best weights (Keras format, 37.8 MB)
│   └── class_names.json          # Index-to-class mapping dictionary
├── results/
│   ├── accuracy_curve.png        # Training vs validation accuracy curve
│   ├── loss_curve.png            # Training vs validation loss curve
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── sample_predictions.png    # Visualization of test sample predictions
│   ├── metrics.txt               # Key accuracy and loss summaries
│   ├── classification_report.txt # Per-class precision, recall, and F1
│   └── history.json              # Per-epoch training logs
├── scripts/
│   ├── prepare_dataset.py        # Dataset downloader and stratified splitter
│   └── verify_pipeline_and_model.py # Sanity check for data and model graph
├── src/
│   ├── data_pipeline.py          # tf.data dataset creation utility
│   ├── model.py                  # CNN model architecture definition
│   ├── train.py                  # Training pipeline with ModelCheckpoint
│   ├── evaluate.py               # Evaluation script loading saved model
│   ├── predict.py                # Standalone CLI prediction module
│   └── app.py                    # Standalone HTTP live model web demo
├── plan.md                       # Architectural design specification
├── README.md                     # Comprehensive project documentation
└── requirements.txt              # Pinned library dependencies
```

---

## 8. Setup & Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment (`venv`)

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd ~/Desktop/DL-Project

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

### 2. Verify Data Pipeline & Model Architecture
```bash
python scripts/verify_pipeline_and_model.py
```

### 3. Train the Model
```bash
python src/train.py
```
*Saves the best model to `models/best_model.keras` and training logs to `results/history.json`.*

### 4. Evaluate on Test Set
```bash
python src/evaluate.py
```
*Generates all evaluation curves, confusion matrix, and classification report in `results/`.*

---

## 9. CLI Inference

Inference can be run directly from the command line on any JPEG/PNG image:

```bash
.venv/bin/python src/predict.py <image_path>
```

### Examples:
```bash
# Predict a test Juice image:
.venv/bin/python src/predict.py data/test/Juice/Bravo-Apple-Juice_005.jpg
# Output:
# Predicted class: Juice
# Confidence: 83.6%

# Predict a test Milk image:
.venv/bin/python src/predict.py data/test/Milk/Arla-Ecological-Medium-Fat-Milk_010.jpg
# Output:
# Predicted class: Milk
# Confidence: 67.3%

# Predict a test Yoghurt image:
.venv/bin/python src/predict.py data/test/Yoghurt/Arla-Mild-Vanilla-Yoghurt_020.jpg
# Output:
# Predicted class: Yoghurt
# Confidence: 97.5%
```

---

## 10. Live Model Demo (Web Application)

The project includes a self-contained local web server built with Python's standard library `http.server`. It requires no external web frameworks (no Flask, FastAPI, or Streamlit).

### Start the Live Server
```bash
.venv/bin/python src/app.py
```
Output:
```
Initializing Kirana Product Recognition Engine...
Loading trained model from /home/asus/Desktop/DL-Project/models/best_model.keras...
Model loaded successfully with classes: ['Juice', 'Milk', 'Yoghurt']
=================================================================
Kirana Product Recognition Web UI active!
URL: http://127.0.0.1:5000
Press Ctrl+C to stop.
=================================================================
```

### Accessing the Web Demo
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- **Image Upload:** Drag and drop or browse any shelf product image (`.jpg`, `.png`).
- **Live Inference:** Click "Classify Product". The browser sends the raw image to the `/predict` endpoint.
- **Visual Feedback:** Shows predicted class, confidence bar, and full probability distribution across all 3 classes.

### API Endpoints
- `GET /` — Serves the responsive, modern Web UI.
- `GET /info` — Returns JSON metadata about the model, classes, and accuracy.
- `POST /predict` — Accepts image bytes (raw or multipart/form-data) and runs live inference using `best_model.keras`.

---

## 11. Viva Voce Preparation Guide

### Q1: Why use a CNN instead of a standard Multi-Layer Perceptron (Dense NN) for images?
> **Answer:** Fully-connected Dense layers ignore 2D spatial locality and require flattening images into 1D vectors, resulting in millions of redundant parameters. CNNs use shared convolutional filters that exploit translation invariance and spatial locality, extracting local visual features (edges, corners, patterns) regardless of where they appear on the shelf.

### Q2: What is the role of MaxPooling in your architecture?
> **Answer:** MaxPooling downsamples spatial dimensions (height and width) by taking the maximum value in each $2 \times 2$ window. This reduces computational complexity and parameter count, controls overfitting, and introduces translation invariance (the feature remains recognized even if shifted slightly).

### Q3: Why is `Rescaling` placed inside the model instead of pre-processing images outside?
> **Answer:** Embedding `layers.Rescaling(1/255.0)` directly into the Keras model makes the model completely self-contained. During production, inference, or Web Demo deployment, callers can feed raw RGB pixel values $[0, 255]$ without risking normalization bugs or distribution mismatches.

### Q4: Why use `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?
> **Answer:** Our labels are represented as single integer class indices ($0, 1, 2$) rather than one-hot encoded vectors ($[1, 0, 0]$, $[0, 1, 0]$, etc.). `sparse_categorical_crossentropy` operates directly on integer labels, saving memory without needing one-hot encoding.

### Q5: Why is `shuffle=False` critical for validation and test datasets?
> **Answer:** During evaluation, predictions are generated sequentially batch by batch. If the test set were shuffled, the order of true labels extracted from the dataset would not correspond to the sequential prediction indices, causing completely distorted confusion matrices and classification reports.

### Q6: How does Dropout prevent overfitting?
> **Answer:** `Dropout(0.5)` randomly deactivates 50% of the neurons in the dense layer during each training forward pass. This forces the network to learn redundant, robust feature representations rather than co-adapting and memorizing individual training sample quirks. Dropout is automatically deactivated during evaluation and inference.

### Q7: How does this classification system relate to real Kirana store inventory automation?
> **Answer:** In a real store, an object detector (e.g., YOLO) would first detect and crop each bounding box representing an item on the shelf. This CNN classifier would then classify the cropped box into its product category (`Juice`, `Milk`, `Yoghurt`) to update inventory counts automatically.
