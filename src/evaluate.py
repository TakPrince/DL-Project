"""
Final Evaluation Module for Packaged Grocery Classification.
Loads models/best_model.keras and evaluates on data/test/.
Generates all curves, reports, matrices, and sample visualizations.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Use Agg backend for headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path("/home/asus/Desktop/DL-Project")
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

def plot_curves(history_data: dict, results_dir: Path):
    """Plots and saves accuracy and loss curves."""
    epochs = range(1, len(history_data["history"]["loss"]) + 1)
    best_epoch = history_data["best_epoch"]
    
    # 1. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_data["history"]["accuracy"], "b-o", label="Training Accuracy", markersize=4)
    plt.plot(epochs, history_data["history"]["val_accuracy"], "g-s", label="Validation Accuracy", markersize=4)
    plt.axvline(x=best_epoch, color="r", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    plt.title("Model Accuracy over Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    acc_path = results_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {acc_path}")
    
    # 2. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_data["history"]["loss"], "b-o", label="Training Loss", markersize=4)
    plt.plot(epochs, history_data["history"]["val_loss"], "g-s", label="Validation Loss", markersize=4)
    plt.axvline(x=best_epoch, color="r", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    plt.title("Model Loss over Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    loss_path = results_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {loss_path}")

def plot_confusion_matrix(cm: np.ndarray, class_names: list, results_dir: Path):
    """Plots and saves confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    
    # Annotate counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold"
            )
            
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_title("Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")

def plot_sample_predictions(test_ds, model, class_names: list, results_dir: Path):
    """Plots a 3x3 grid of sample test predictions with confidence."""
    images_list = []
    labels_list = []
    for imgs, lbls in test_ds:
        images_list.append(imgs.numpy())
        labels_list.append(lbls.numpy())
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    preds_prob = model.predict(images, verbose=0)
    preds_cls = np.argmax(preds_prob, axis=1)
    
    # Select 9 diverse indices
    num_samples = min(9, len(images))
    indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        img = images[idx].astype("uint8")
        true_lbl = class_names[labels[idx]]
        pred_lbl = class_names[preds_cls[idx]]
        conf = preds_prob[idx][preds_cls[idx]] * 100.0
        
        ax.imshow(img)
        ax.axis("off")
        color = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(
            f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)",
            color=color, fontsize=11, fontweight="bold"
        )
        
    plt.suptitle("Sample Test Predictions (Baseline CNN)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    sample_path = results_dir / "sample_predictions.png"
    plt.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {sample_path}")

def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load class names
    class_names_file = MODELS_DIR / "class_names.json"
    with open(class_names_file, "r", encoding="utf-8") as f:
        class_meta = json.load(f)
    class_names = class_meta["class_names"]
    print(f"Loaded class names from disk: {class_names}")
    
    # 2. Load trained model FROM DISK
    model_path = MODELS_DIR / "best_model.keras"
    print(f"Loading best model from disk: {model_path}")
    model = keras.models.load_model(model_path)
    
    # 3. Load history.json for curves
    history_file = RESULTS_DIR / "history.json"
    with open(history_file, "r", encoding="utf-8") as f:
        history_data = json.load(f)
        
    # 4. Generate curves
    plot_curves(history_data, RESULTS_DIR)
    
    # 5. Load test dataset (strictly shuffle=False, seed=42)
    test_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=(128, 128),
        color_mode="rgb",
        batch_size=32,
        label_mode="int",
        shuffle=False,
        seed=42
    )
    
    # 6. Evaluate loss and accuracy on test dataset
    print("\nRunning model.evaluate() on test set...")
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss = float(eval_results[0])
    test_acc = float(eval_results[1])
    
    # 7. Collect all predictions and true labels for classification report & confusion matrix
    y_true_list = []
    y_pred_probs_list = []
    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(lbls.numpy())
        
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 8. Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    clf_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save classification report
    clf_report_path = RESULTS_DIR / "classification_report.txt"
    with open(clf_report_path, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION REPORT (TEST SET) ===\n")
        f.write(clf_report)
    print(f"Saved: {clf_report_path}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, RESULTS_DIR)
    
    # Plot sample predictions
    plot_sample_predictions(test_ds, model, class_names, RESULTS_DIR)
    
    # Best train/val from history
    best_epoch = history_data["best_epoch"]
    best_val_loss = history_data["best_val_loss"]
    best_val_acc = history_data["best_val_accuracy"]
    best_train_acc = max(history_data["history"]["accuracy"])
    
    chance_acc = 1.0 / len(class_names)
    
    # Save metrics.txt
    metrics_path = RESULTS_DIR / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== FINAL MODEL EVALUATION METRICS ===\n")
        f.write(f"Classes:                  {class_names}\n")
        f.write(f"Chance Accuracy:          {chance_acc * 100:.2f}%\n")
        f.write(f"Best Training Accuracy:   {best_train_acc:.4f} ({best_train_acc * 100:.2f}%)\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)\n")
        f.write(f"Best Validation Loss:     {best_val_loss:.4f} (at Epoch {best_epoch})\n")
        f.write(f"Test Accuracy:            {test_acc:.4f} ({test_acc * 100:.2f}%)\n")
        f.write(f"Test Loss:                {test_loss:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        for cls in class_names:
            m = clf_dict[cls]
            f.write(f"  {cls:<10} - Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1-score: {m['f1-score']:.4f}, Support: {m['support']}\n")
        f.write("\nMacro Average:\n")
        macro = clf_dict["macro avg"]
        f.write(f"  Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f}, F1-score: {macro['f1-score']:.4f}\n")
        f.write("\nWeighted Average:\n")
        weighted = clf_dict["weighted avg"]
        f.write(f"  Precision: {weighted['precision']:.4f}, Recall: {weighted['recall']:.4f}, F1-score: {weighted['f1-score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"Rows = True, Columns = Pred: {class_names}\n")
        for row in cm:
            f.write(f"  {row.tolist()}\n")
            
    print(f"Saved: {metrics_path}")
    
    # Print console output
    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Chance Baseline Accuracy: {chance_acc * 100:.2f}%")
    print(f"Best Training Accuracy:   {best_train_acc:.4f} ({best_train_acc * 100:.2f}%)")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Test Accuracy:            {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Test Loss:                {test_loss:.4f}")
    print("\nClassification Report:\n", clf_report)
    print("Confusion Matrix:\n", cm)
    print("=" * 65)

if __name__ == "__main__":
    evaluate()
