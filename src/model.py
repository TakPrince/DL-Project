"""
Baseline CNN Architecture for Packaged Grocery Classification.
Follows PLAN.md specifications strictly.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_baseline_cnn(input_shape: tuple = (128, 128, 3), num_classes: int = 3, learning_rate: float = 0.001):
    """
    Builds and compiles the baseline CNN with integrated augmentation and rescaling layers.
    """
    tf.keras.utils.set_random_seed(42)
    
    model = keras.Sequential([
        # Input Layer
        keras.Input(shape=input_shape, name="input_image"),
        
        # In-model Augmentation Layers
        layers.RandomRotation(0.05, seed=42, name="aug_random_rotation"),
        layers.RandomZoom(0.05, seed=42, name="aug_random_zoom"),
        layers.RandomFlip("horizontal", seed=42, name="aug_random_flip"),
        
        # Normalization
        layers.Rescaling(1.0 / 255.0, name="rescaling_norm"),
        
        # Feature Extraction Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid", name="conv2d_block1"),
        layers.MaxPooling2D((2, 2), name="maxpool_block1"),
        
        # Feature Extraction Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv2d_block2"),
        layers.MaxPooling2D((2, 2), name="maxpool_block2"),
        
        # Feature Extraction Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="valid", name="conv2d_block3"),
        layers.MaxPooling2D((2, 2), name="maxpool_block3"),
        
        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_dense128"),
        layers.Dropout(0.5, seed=42, name="dropout_regularization"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="baseline_grocery_cnn")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
