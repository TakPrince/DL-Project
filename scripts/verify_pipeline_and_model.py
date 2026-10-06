#!/usr/bin/env python3
"""
Verification Script for Preprocessing Pipeline and Baseline CNN.
Follows PLAN.md specifications strictly.
Does NOT train the model.
"""

import sys
import json
from pathlib import Path
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path("/home/asus/Desktop/DL-Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import create_datasets
from src.model import build_baseline_cnn

def main():
    print("==================================================================")
    print("Part A: Preprocessing Pipeline Verification")
    print("==================================================================")
    
    # 1. Create datasets
    train_ds, val_ds, test_ds, class_names, class_map = create_datasets()
    
    # 2. Inspect class names and mapping
    print(f"\nClass names: {class_names}")
    print(f"Class-to-label mapping: {class_map}")
    
    # Verify saved class_names.json
    class_names_path = PROJECT_ROOT / "models" / "class_names.json"
    with open(class_names_path, "r", encoding="utf-8") as f:
        saved_info = json.load(f)
    print(f"Verified {class_names_path} exists and contains: {saved_info}")
    
    # 3. Take one batch from train_ds
    for images_batch, labels_batch in train_ds.take(1):
        raw_min = tf.reduce_min(images_batch).numpy()
        raw_max = tf.reduce_max(images_batch).numpy()
        print(f"\nBatch inspection:")
        print(f"  Images batch shape: {images_batch.shape}")
        print(f"  Labels batch shape: {labels_batch.shape}")
        print(f"  Labels sample: {labels_batch[:8].numpy()}")
        print(f"  Pixel range BEFORE model preprocessing: min = {raw_min:.2f}, max = {raw_max:.2f}")
        break
        
    print("\n==================================================================")
    print("Part B: Baseline CNN Architecture Construction & Verification")
    print("==================================================================")
    
    # 4. Build and compile baseline CNN
    model = build_baseline_cnn(input_shape=(128, 128, 3), num_classes=len(class_names))
    
    # 5. Verify preprocessing inside model
    # Pass the images through the rescaling layer
    rescaling_layer = model.get_layer("rescaling_norm")
    rescaled_images = rescaling_layer(images_batch)
    norm_min = tf.reduce_min(rescaled_images).numpy()
    norm_max = tf.reduce_max(rescaled_images).numpy()
    print(f"\nIn-model Preprocessing Check:")
    print(f"  Pixel range AFTER Rescaling(1/255): min = {norm_min:.4f}, max = {norm_max:.4f}")
    
    # 6. Test forward pass inference on the batch (no training)
    predictions = model(images_batch, training=False)
    print(f"  Forward pass prediction shape: {predictions.shape}")
    print(f"  Sum of probabilities per sample (softmax check): {tf.reduce_sum(predictions[0]).numpy():.4f}")
    
    # 7. Print Model Summary
    print("\nModel Summary:")
    model.summary()
    
    # 8. Parameter counts
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable_params = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    
    print("\nParameter Summary:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Non-trainable params: {non_trainable_params:,}")
    
    print("\n[SUCCESS] Pipeline and model verified successfully. Model was NOT trained.")

if __name__ == "__main__":
    main()
