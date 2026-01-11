#!/usr/bin/env python3
"""
evaluate.py - Validation Set Evaluation
Usage:
  ./evaluate.py <path_to_dataset_directory>
"""

import sys
import os
import json
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import Transformation

IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'best_leaf_model.keras'
CLASSES_PATH = 'classes.json'


def load_resources():
    """Load model and classes"""
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error : Model not found ({MODEL_PATH})")

    class_names = []
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, class_names
    except Exception as e:
        sys.exit(f"Fail loading files: {e}")


def preprocess_image(image_path):
    """
    Apply exactly the same preprocessing as Predict.py
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_resized_bgr = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))

    transformer = Transformation.Transformation(img_resized_bgr)
    img_transformed_bgr = transformer.masked_leaf()

    img_transformed_rgb = cv2.cvtColor(img_transformed_bgr, cv2.COLOR_BGR2RGB)

    return img_transformed_rgb


def evaluate_dataset(dataset_path, model, class_names):
    correct_predictions = 0
    total_images = 0

    if not class_names:
        class_names = sorted(
            [d for d in os.listdir(dataset_path)
             if os.path.isdir(os.path.join(dataset_path, d))])
        print(f"Found Classes: {class_names}")

    print(f"Starting evaluation : {dataset_path}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = os.listdir(class_dir)
        print(f"Processing class : '{class_name}' "
              f"({len(files)} images)...")

        for file_name in tqdm(files):
            file_path = os.path.join(class_dir, file_name)

            if not file_name.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            processed_img = preprocess_image(file_path)
            if processed_img is None:
                continue

            img_batch = np.expand_dims(processed_img, axis=0)
            predictions = model.predict(img_batch, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_label = class_names[predicted_index]

            if predicted_label == class_name:
                correct_predictions += 1

            total_images += 1

    return correct_predictions, total_images


def main():
    if len(sys.argv) < 2:
        print("Usage: ./evaluate.py <path_to_dataset_root>")
        print("Example: ./evaluate.py ./Apple_Grape_Potato/Validation")
        sys.exit(1)

    dataset_path = sys.argv[1]

    model, class_names = load_resources()

    correct, total = evaluate_dataset(dataset_path, model, class_names)

    if total == 0:
        print("Aucune image trouvée.")
        sys.exit(1)

    accuracy = correct / total

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total images   : {total}")
    print(f"Good predicts: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 40)


if __name__ == "__main__":
    main()
