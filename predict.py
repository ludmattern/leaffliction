#!/usr/bin/env python3
"""
Predict.py - Classification de feuilles
Charge un modèle entraîné, transforme l'image d'entrée et prédit la maladie.

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

# Import du module local Transformation
try:
    import Transformation
except ImportError:
    sys.exit("Erreur : Le fichier Transformation.py doit être dans le même dossier.")

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'best_leaf_model.keras'  # Ou .h5 selon votre choix dans train.py
CLASSES_PATH = 'classes.json'


def load_resources():
    """Charge le modèle et les noms de classes."""
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Erreur : Modèle introuvable ({MODEL_PATH}). Avez-vous lancé train.py ?")

    if not os.path.exists(CLASSES_PATH):
        sys.exit(f"Erreur : Fichier de classes introuvable ({CLASSES_PATH}).")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        sys.exit(f"Erreur lors du chargement des ressources : {e}")


def predict_image(image_path, model, class_names):
    """Prépare l'image, applique la transformation et prédit."""
    # 1. Chargement (OpenCV lit en BGR par défaut)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit("Erreur : Impossible de lire l'image.")

    # 2. Redimensionnement (On garde le format BGR pour la transformation)
    # On redimensionne avant la transformation pour gagner en performance
    img_resized_bgr = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))

    # 3. Transformation (Masquage)
    # La classe Transformation s'attend à du BGR (format OpenCV standard)
    transformer = Transformation.Transformation(img_resized_bgr)
    img_transformed_bgr = transformer.masked_leaf()

    # 4. Préparation pour le modèle (Conversion BGR -> RGB)
    # Le modèle a été entraîné avec Keras qui charge en RGB
    img_transformed_rgb = cv2.cvtColor(img_transformed_bgr, cv2.COLOR_BGR2RGB)

    # Ajout de la dimension batch (1, 128, 128, 3)
    img_batch = np.expand_dims(img_transformed_rgb, axis=0)

    # 5. Prédiction
    predictions = model.predict(img_batch, verbose=0)
    score_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_label = class_names[score_index]

    # Pour l'affichage, on a aussi besoin de l'original en RGB
    img_original_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)

    return predicted_label, confidence, img_original_rgb, img_transformed_rgb


def display_result(original, transformed, label, confidence):
    """Affiche le résultat avec Matplotlib."""

    # Configuration esthétique sombre
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Prediction: {label}', fontsize=16, color='#4CAF50', fontweight='bold')

    # Image Originale
    ax1.imshow(original)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Image Transformée (Celle vue par le réseau de neurones)
    ax2.imshow(transformed)
    ax2.set_title("Transformed (Masked) Image")
    ax2.axis('off')

    # Texte de confiance en bas
    plt.figtext(0.5, 0.05, f"Confidence: {confidence:.2%}",
                ha="center", fontsize=12, color="white")

    plt.show()


def main():
    # Vérification des arguments
    if len(sys.argv) < 2:
        print("Usage: ./predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Chargement
    model, class_names = load_resources()

    # Prédiction
    label, confidence, img_orig, img_trans = predict_image(image_path, model, class_names)

    # [cite_start]Affichage Terminal (Format demandé par le sujet [cite: 210-212])
    print("DL classification")
    print(f"Class predicted: {label}")

    # Affichage Graphique
    display_result(img_orig, img_trans, label, confidence)


if __name__ == "__main__":
    main()