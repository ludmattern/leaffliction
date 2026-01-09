#!/usr/bin/env python3
"""
Predict.py - Leaf Classification
Usage:
  ./predict.py <path_to_image>
"""

import sys
import os
import json
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import Transformation


IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'best_leaf_model.keras'
CLASSES_PATH = 'classes.json'


def load_resources():
    """Load ;odel and classes"""
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error : Model not found ({MODEL_PATH})")

    if not os.path.exists(CLASSES_PATH):
        sys.exit(f"Error : Classes file not found ({CLASSES_PATH}).")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        sys.exit(f"Fail loading files: {e}")


def predict_image(image_path, model, class_names):
    """Apply transformation to image and predict class"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit("Error : Fail to read image.")

    img_resized_bgr = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))
    transformer = Transformation.Transformation(img_resized_bgr)
    img_transformed_bgr = transformer.masked_leaf()
    img_transformed_rgb = cv2.cvtColor(img_transformed_bgr, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_transformed_rgb, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    score_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_label = class_names[score_index]

    img_original_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)

    return predicted_label, confidence, img_original_rgb, img_transformed_rgb


def display_result(original, transformed, label, confidence):
    """Display result"""

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f'Prediction: {label}',
        fontsize=16, color='#4CAF50', fontweight='bold')

    ax1.imshow(original)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(transformed)
    ax2.set_title("Transformed (Masked) Image")
    ax2.axis('off')

    plt.figtext(0.5, 0.05, f"Confidence: {confidence:.2%}",
                ha="center", fontsize=12, color="white")

    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: ./predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    model, class_names = load_resources()
    label, confidence, img_orig, img_trans = predict_image(
        image_path, model, class_names)
    print("DL classification")
    print(f"Class predicted: {label}")

    display_result(img_orig, img_trans, label, confidence)


if __name__ == "__main__":
    main()
