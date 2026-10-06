# PLAN.md

# Kirana Store Shelf Product Recognition Using CNN

**Student:** Prince Kumar Tak | **ID:** ADT25SOCM0132 | **Unit III** | **CO2, CO4**

This file is the implementation specification for the coding agent (Antigravity). Read it fully before making any change. Follow it in order. Do not add scope. Contains no implementation code.

---

## 1. Goal

Build a small, clean CNN image classifier that recognizes packaged grocery/kirana products from product/shelf images.

**Core deliverable:** CNN-based product image classification:

Dataset → Inspection → Split → Preprocessing → CNN → Training → Evaluation → Prediction

**Out of scope:** object detection, counting, production systems, frontend, API, database, authentication, Docker, cloud deployment. Detection and counting are future extensions only.

**Priorities:** working result → simple → viva-friendly → reproducible → fast to run.

---

## 2. Technology

**Allowed:** Python, TensorFlow/Keras, NumPy, Pillow, Matplotlib, scikit-learn. CLI scripts in `src/` are the default (notebooks optional).

**Not allowed:** any other library unless genuinely necessary; justify any addition in the README. A dataset download tool (e.g., Kaggle CLI) may be used only to fetch the approved subset. No web frameworks, detection frameworks, MLOps tools, or LLM tooling.

---

## 3. Environment Safety Rules

- Use a virtual environment (`venv`) or the already-existing environment. Do **not** install or upgrade packages globally.
- **List the packages you intend to install and wait for approval before installing.**
- Do not delete, move, or overwrite anything outside the project folder.
- Do not modify unrelated files or projects.

---

## 4. Project Structure

```
kirana-product-recognition/
├── dataset/              # train/ validation/ test/ -> one folder per class (do not commit)
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/               # best_model.keras + class_names.json
├── results/              # plots, metrics, history
├── PLAN.md
├── README.md
├── requirements.txt
└── .gitignore            # only if Git is used; must exclude dataset/
```

Do not create more files than this.

---

## 5. Dataset Strategy

### 5.1 Decision order

1. A **small, classification-ready** grocery/product dataset (folders of product images per class).
2. A **small portion** of a larger dataset.
3. If neither is practical, **ask the user**.

### 5.2 Hard rules

- **Propose 2–3 candidate datasets (name, source link, size, structure) and wait for approval before downloading anything.**
- **Hard size cap:** if a download would exceed ~<100–200 MB MB, stop and ask.
- Warn: SKU-110K-style shelf datasets are detection-oriented with one generic "object" class and are **unsuitable** for classification. Do not use them unless product-level class labels exist.
- Never create fake or synthetic data. Do not mix in extra data sources.
- If the original Kaggle dataset is inconvenient, use a smaller equivalent public dataset or prepared subset and state which one in the README.

### 5.3 Size targets

- **Start with 3 classes**; use up to 5 only if data is abundant and balanced.
- **Minimum ~80 images per class**, ideally 100–300 per class. Roughly balanced.
- Note: with a 15% test split, ~100 images/class yields only ~15 test images per class, so results are indicative, not statistically strong. State this in the README.

### 5.4 Inspect first

Inspect the real folder/file structure, file types, and annotation format. **Never assume the structure.**

### 5.5 Cleaning

- Detect and remove unreadable/corrupted images; report how many.
- Convert all images to **RGB** (some product PNGs have alpha channels or are grayscale).

---

## 6. Split and Leakage Prevention

- **Split files on disk first**, before any training pipeline or augmentation.
- Ratio about **70% train / 15% validation / 15% test**, stratified per class, fixed seed 42.
- **If crops are made from shelf photos, split by source image, not by crop**, so crops from the same photo never land in different splits.
- If the dataset has near-identical consecutive frames or duplicates, remove or group them before splitting.
- **Verify zero overlap** between train/validation/test using filenames **and** file hashes. Report the result.
- The test set is used only for final evaluation. Never for training, early stopping, or model selection.

---

## 7. Preprocessing and Augmentation

- **Load** with Keras `image_dataset_from_directory`, image size 128×128, RGB, batch size 32, integer labels.
- **Shuffle:** training set shuffled with fixed seed. **Validation and test sets: `shuffle=False`**, so predictions align with labels and the confusion matrix is correct.
- **Normalization:** a `Rescaling(1/255)` layer **inside the model**.
- **Labels:** integer-encoded from sorted class folder names. Save the class list to `models/class_names.json`; `evaluate.py` and `predict.py` must use this exact mapping.
- **Augmentation:** Keras preprocessing **layers** inside the model (`RandomFlip`, `RandomRotation`, `RandomZoom`), active only during training. Do not use deprecated `ImageDataGenerator`. Never save augmented images to disk.
- Keep augmentation mild (small rotation ~±5–10%, small zoom ~±10%). Drop horizontal flip if mirrored text/logos change product identity, and note it in the README.

---

## 8. TensorFlow/Keras Compatibility (Phase 1 check)

- Record the installed TensorFlow/Keras version in Phase 1.
- The `.keras` model format requires TF/Keras ≥ 2.12. If the installed version is older, use `.h5` (or upgrade with approval).
- After everything runs, **pin the actually-working versions** of the five libraries in `requirements.txt`.
- Call `tf.keras.utils.set_random_seed(42)` at the top of training. Exact bit-for-bit reproducibility (especially on GPU) is not guaranteed; results should be "close," and the README should say so.

---

## 9. CNN Architecture

Simple and explainable. Adjust filter counts or Dense units only if the dataset requires it.

```
Input (128×128×3)
→ Augmentation layers (training only)
→ Rescaling (1/255)
→ Conv2D (32, 3×3) + ReLU → MaxPooling (2×2)
→ Conv2D (64, 3×3) + ReLU → MaxPooling (2×2)
→ Conv2D (128, 3×3) + ReLU → MaxPooling (2×2)
→ Flatten
→ Dense (128) + ReLU
→ Dropout (0.5)
→ Dense (num_classes) + Softmax
```

- **Loss:** sparse categorical crossentropy (integer labels)
- **Optimizer:** Adam, learning rate 0.001
- **Metric:** accuracy
- Keep `Flatten` (simplest to explain in viva). Overfitting is **expected** on a small dataset (the Dense layer after Flatten holds ~4M parameters). The README must discuss it using the accuracy/loss curves.
- Print `model.summary()` to the terminal and paste it in the README. No separate summary file.

---

## 10. Training Configuration

| Setting        | Value                                                       |
| -------------- | ----------------------------------------------------------- |
| Batch size     | 32                                                          |
| Max epochs     | 20                                                          |
| Early stopping | monitor`val_loss`, patience 4, restore best weights         |
| Checkpoint     | save best model (by`val_loss`) to `models/best_model.keras` |
| Seed           | 42                                                          |

- **Save training history to `results/history.json`** (per-epoch train/val accuracy and loss). `evaluate.py` is a separate script and plots from this file.
- Do not train longer than needed.

---

## 11. Evaluation

All numbers must come from **actual model execution**. Never invent, estimate, or hand-enter results.

`evaluate.py` must:

- Load **`models/best_model.keras` from disk** (not an in-memory model).
- Use the **unshuffled** test set and the saved `class_names.json`.

| Output                                                            | File                                |
| ----------------------------------------------------------------- | ----------------------------------- |
| Training and validation accuracy graph                            | `results/accuracy_curve.png`        |
| Training and validation loss graph                                | `results/loss_curve.png`            |
| Train, validation, test accuracy + test loss                      | `results/metrics.txt`               |
| Confusion matrix                                                  | `results/confusion_matrix.png`      |
| Classification report (precision, recall, F1, support)            | `results/classification_report.txt` |
| Sample test predictions (true label, predicted label, confidence) | `results/sample_predictions.png`    |

**Metric definitions (state in README):**

- Train and validation accuracy: values from `history.json` at the best epoch. Training accuracy is measured with dropout and augmentation active, so it can look lower than expected.
- Test accuracy: a fresh evaluation of the best model on the test set.

**Sanity check:** if test accuracy is near or below chance (1 / num_classes + small margin), treat it as a **bug** (label misalignment, shuffled test set, double normalization, wrong class mapping), not a result. Fix before continuing.

---

## 12. Prediction Script

`src/predict.py` takes an image path as a command-line argument and prints:

```
Predicted class: <class_name>
Confidence: <xx.x>%
```

- Load the saved model and `class_names.json`.
- Load as **RGB** (convert RGBA/grayscale), resize to 128×128, `float32`. Do **not** normalize (Rescaling is inside the model).
- Show a clear error if the path is missing or the file is not a valid image.
- Keep it short and easy to demo.

---

## 13. Development Workflow

Work incrementally. **Test each stage before moving to the next.** Fix actual errors instead of rewriting working code.

1. **Inspect.** Working directory, Python and TF/Keras versions, installed packages, CPU/GPU, dataset availability and structure.
   **STOP and report to the user:** findings, 2–3 candidate datasets, proposed classes, intended package installs. **Wait for approval before downloading or installing anything.**
2. **Prepare dataset.** Build the subset, clean images, split (Section 6), run the leakage check. Report class names, counts per class, removed images, split sizes.
3. **Preprocessing pipeline.** Load with Section 7 settings. Confirm batch shape (32, 128, 128, 3) and that val/test are unshuffled.
4. **Implement CNN.** Confirm it builds and summary prints.
5. **Train.** Run real training. Confirm best model and `history.json` are saved.
6. **Evaluate.** Generate all outputs in Section 11 and apply the sanity check.
7. **Prediction script.** Run on 2–3 test images and confirm the output format.
8. **Verify results.** Confirm all files exist in `results/` and numbers are real.
9. **README.md.** Write actual dataset name/source, classes and counts, preprocessing, architecture, training config, real results, overfitting discussion, small-test-set note, reproducibility note, and run instructions. `requirements.txt` lists only libraries actually used, with pinned versions.

---

## 14. If Accuracy Is Poor (below ~70%)

First rule out bugs (Section 11 sanity check). Then escalate **in this order only**:

1. Mildly stronger augmentation
2. More epochs (maximum 30)
3. Slightly more images per class
4. **Stop and ask the user** about transfer learning

Do not skip steps or switch to a different ML approach on your own.

---

## 15. Agent Rules

- Read PLAN.md before making changes.
- Inspect before assuming; never assume dataset structure.
- Ask for approval before downloads and installs; respect the 500 MB cap.
- Keep the implementation minimal and beginner-friendly, with useful but not excessive comments.
- Avoid unnecessary dependencies and over-engineering.
- Run and verify code after each major change.
- Do not build a frontend, API, database, authentication, Docker setup, cloud deployment, or production infrastructure.
- Do not add object detection or counting.
- Do not use transfer learning without user approval.
- Do not invent accuracy or results.
- Stop once the completion criteria are met.

---

## 16. Completion Criteria

- [ ] Dataset approved, loads correctly, classes identified correctly
- [ ] Images valid and RGB; split works; zero overlap verified; classes reasonably balanced
- [ ] CNN builds, trains successfully, best model saved, `history.json` saved
- [ ] Evaluation loads the best model from disk and uses an unshuffled test set
- [ ] Accuracy/loss graphs, confusion matrix, classification report, and sample predictions generated
- [ ] Sanity check passed (test accuracy clearly above chance)
- [ ] `predict.py` accepts a new image and returns class and confidence
- [ ] Reproducibility: fixed seed, pinned versions, documented configuration
- [ ] README explains how to run the project and contains actual results only
- [ ] Code is simple enough to explain in a college viva

---

## 17. Viva Preparation (Student Only, Not Implementation Tasks)

**CNN basics:** What is a CNN and why use it for images? Convolution, filter/kernel, ReLU, pooling and why MaxPooling, Flatten, Dense layer, Softmax.

**Training:** Epoch, batch size, learning rate, loss function (why sparse categorical crossentropy), why Adam, overfitting and how Dropout helps, data augmentation (and why mild here), early stopping.

**Data and evaluation:** Train vs validation vs test and why separate, data leakage (why split by source image), confusion matrix, accuracy, precision, recall, F1-score, reading curves to spot overfitting, why train accuracy can be lower than validation accuracy with Dropout/augmentation.

**Project-level:** Why classification instead of detection? How was the subset selected? How could this support inventory counting? (Detect each product on the shelf, classify each box with this CNN, count per class.)

---

## 18. Documentation Roles

- **PLAN.md:** implementation specification for Antigravity (this file).
- **README.md:** final documentation, written after implementation, with real results.

---

## 19. Future Scope (Not to Be Implemented)

Object detection, product counting, shelf-level detection, real-time camera input, larger product catalogs, OCR/barcode integration, inventory management integration.
