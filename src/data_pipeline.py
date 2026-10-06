"""
Data Pipeline Module for Packaged Grocery Classification.
Uses tf.keras.utils.image_dataset_from_directory with deterministic configuration.
"""

import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

def create_datasets(
    data_dir: Path = Path("/home/asus/Desktop/DL-Project/data"),
    models_dir: Path = Path("/home/asus/Desktop/DL-Project/models"),
    image_size: tuple = (128, 128),
    batch_size: int = 32,
    seed: int = 42
):
    """
    Creates train, validation, and test datasets from directory splits.
    Saves class names mapping to models/class_names.json.
    """
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training Dataset (shuffled, seed=42)
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=seed
    )
    
    # 2. Validation Dataset (not shuffled, seed=42)
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
        seed=seed
    )
    
    # 3. Test Dataset (not shuffled, seed=42)
    test_ds = keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
        seed=seed
    )
    
    class_names = sorted(train_ds.class_names)
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    # Save class_names.json
    class_names_path = models_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "class_to_idx": class_map,
            "num_classes": len(class_names)
        }, f, indent=2)
    print(f"Saved class names mapping to: {class_names_path}")
    
    return train_ds, val_ds, test_ds, class_names, class_map
