"""
Inference Module for Packaged Grocery Classification.
Adheres strictly to the verified model preprocessing pipeline.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import io
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path("/home/asus/Desktop/DL-Project")
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH, class_names_path: Path = CLASS_NAMES_PATH, verbose: bool = True):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.verbose = verbose
        self.model = None
        self.class_names = []
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file missing: {self.model_path}")
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class mapping file missing: {self.class_names_path}")
            
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.class_names = meta["class_names"]
            
        if self.verbose:
            print(f"Loading trained model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        if self.verbose:
            print(f"Model loaded successfully with classes: {self.class_names}")

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs prediction on raw image bytes.
        Preprocessing:
        1. Open with PIL
        2. Convert to RGB
        3. Resize to 128x128
        4. Convert to float32 NumPy array
        5. Add batch dimension
        6. Do NOT divide by 255 (Rescaling is in-model)
        """
        if not image_bytes:
            return {"success": False, "error": "No image data received."}

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
        except Exception:
            return {"success": False, "error": "Invalid or corrupt image format. Supported formats: JPEG, PNG, WEBP."}

        try:
            # Reopen after verify
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
                
            pil_img = pil_img.resize((128, 128), Image.Resampling.BILINEAR)
            img_arr = np.array(pil_img, dtype=np.float32)
            input_tensor = np.expand_dims(img_arr, axis=0)  # Shape: (1, 128, 128, 3)

            # In-model inference
            predictions = self.model.predict(input_tensor, verbose=0)
            probs = predictions[0]

            best_idx = int(np.argmax(probs))
            predicted_class = self.class_names[best_idx]
            confidence_pct = round(float(probs[best_idx] * 100.0), 2)

            probabilities = {
                cls_name: round(float(probs[i] * 100.0), 2)
                for i, cls_name in enumerate(self.class_names)
            }

            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence_pct,
                "probabilities": probabilities,
                "classes": self.class_names
            }
        except Exception as e:
            return {"success": False, "error": f"Prediction execution failed: {str(e)}"}

    def predict_image_path(self, image_path: Path | str) -> Dict[str, Any]:
        """
        Runs prediction on an image file from disk.
        """
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"Image file not found: {path}"}
        if not path.is_file():
            return {"success": False, "error": f"Path is not a regular file: {path}"}
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            return self.predict_bytes(raw_bytes)
        except Exception as e:
            return {"success": False, "error": f"Failed to read image file: {str(e)}"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        predictor = Predictor(verbose=False)
        result = predictor.predict_image_path(image_path)
        if not result.get("success"):
            print(f"Error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)

        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.1f}%")
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
