"""
Training Script for Baseline CNN on Grocery Classification.
Follows PLAN.md specifications strictly.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set path
PROJECT_ROOT = Path("/home/asus/Desktop/DL-Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import create_datasets
from src.model import build_baseline_cnn

def train():
    tf.keras.utils.set_random_seed(42)
    
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = models_dir / "best_model.keras"
    history_path = results_dir / "history.json"
    
    print("=" * 65)
    print("Loading datasets (train & val only)...")
    print("=" * 65)
    train_ds, val_ds, test_ds, class_names, class_map = create_datasets()
    # Explicitly do NOT use test_ds during training
    del test_ds
    
    print("\n" + "=" * 65)
    print("Building baseline CNN model...")
    print("=" * 65)
    model = build_baseline_cnn(input_shape=(128, 128, 3), num_classes=len(class_names), learning_rate=0.001)
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    print("\n" + "=" * 65)
    print("Starting training (max epochs: 20, batch size: 32)...")
    print("=" * 65)
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    training_duration = time.time() - start_time
    
    hist_dict = history.history
    epochs_completed = len(hist_dict["loss"])
    best_epoch_idx = int(np.argmin(hist_dict["val_loss"]))
    best_epoch = best_epoch_idx + 1
    
    best_val_loss = float(hist_dict["val_loss"][best_epoch_idx])
    best_val_acc = float(hist_dict["val_accuracy"][best_epoch_idx])
    final_train_acc = float(hist_dict["accuracy"][-1])
    final_val_acc = float(hist_dict["val_accuracy"][-1])
    
    # Save history to results/history.json
    results_payload = {
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        "training_time_seconds": round(training_duration, 2),
        "history": {k: [float(v) for v in vals] for k, vals in hist_dict.items()}
    }
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved training history to: {history_path}")
    
    model_size_mb = best_model_path.stat().st_size / (1024 * 1024) if best_model_path.exists() else 0.0
    
    print("\n" + "=" * 65)
    print("TRAINING RUN SUMMARY")
    print("=" * 65)
    print(f"Epochs completed:       {epochs_completed}")
    print(f"Best epoch:             {best_epoch}")
    print(f"Best validation loss:   {best_val_loss:.4f}")
    print(f"Best validation acc:    {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Final training acc:     {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"Final validation acc:   {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print(f"Total training time:    {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")
    print(f"Model file size:        {model_size_mb:.2f} MB")
    print(f"Saved best model path:  {best_model_path}")
    print("=" * 65)

if __name__ == "__main__":
    train()
