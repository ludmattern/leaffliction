#!/usr/bin/env python3
"""
Script d'entraînement de classification de feuilles.
Gère le pipeline complet : Distribution -> Augmentation -> Transformation -> Entraînement CNN -> Export.
"""

import sys
import shutil
import json
import zipfile
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

try:
    import Distribution
    import Augmentation
    import Transformation
except ImportError as e:
    sys.exit(f"Erreur critique : Impossible d'importer les modules locaux. {e}")

import warnings
# Ignore les FutureWarning venant de bibliothèques tierces
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TrainConfig:
    """Configuration des hyperparamètres et constantes."""
    img_height: int = 128
    img_width: int = 128
    batch_size: int = 64
    epochs: int = 100
    validation_split: float = 0.2
    seed: int = 123
    augmented_dir: Path = Path("./augmented_dataset")
    output_dir: Path = Path("./masked_dataset")
    stats_file: str = "stats.json"
    classes_file: str = "classes.json"
    model_filename: str = "best_leaf_model.keras"
    zip_filename: str = "learnings.zip"
    plot_filename: str = "training_plot.png"


def validate_arguments() -> Path:
    """Vérifie et retourne le chemin du dossier d'images fourni en argument."""
    if len(sys.argv) < 2:
        logging.error("Usage : ./train.py <chemin_du_dossier>")
        sys.exit(1)

    dossier_images = Path(sys.argv[1])
    if not dossier_images.exists():
        logging.error(f"Le dossier '{dossier_images}' n'existe pas.")
        sys.exit(1)

    return dossier_images


def prepare_directories(config: TrainConfig) -> None:
    """Nettoie et prépare les répertoires de travail."""
    for directory in [config.augmented_dir, config.output_dir]:
        if directory.exists():
            logging.info(f"Nettoyage du répertoire : {directory}")
            shutil.rmtree(directory)


def run_preprocessing_pipeline(source_dir: Path, config: TrainConfig) -> None:
    """
    Exécute les scripts externes de pré-traitement.
    Note : Modifie sys.argv pour s'adapter aux scripts externes existants.
    """
    logging.info("--- Étape 1 : Distribution ---")
    # Sauvegarde des arguments originaux
    original_argv = sys.argv

    try:
        sys.argv = ["Distribution.py", str(source_dir), "--export", config.stats_file, "-np"]
        Distribution.main()

        logging.info("--- Étape 2 : Augmentation ---")
        sys.argv = ["Augmentation.py", "--balance", config.stats_file, "--output", str(config.augmented_dir)]
        Augmentation.main()

        logging.info("--- Étape 3 : Transformation (Masquage) ---")
        Transformation.process_directory(
            src_dir=str(config.augmented_dir),
            dst_dir=str(config.output_dir),
            mask_only=True,
            silent=True
        )
    finally:
        # Restauration des arguments (bonne pratique)
        sys.argv = original_argv


def create_datasets(config: TrainConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Charge et optimise les datasets d'entraînement et de validation."""
    logging.info("Chargement des datasets...")

    # Configuration commune
    ds_params = {
        "directory": config.output_dir,
        "validation_split": config.validation_split,
        "seed": config.seed,
        "image_size": (config.img_height, config.img_width),
        "batch_size": config.batch_size,
        "label_mode": 'categorical'
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **ds_params)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **ds_params)

    class_names = train_ds.class_names
    logging.info(f"Classes détectées : {class_names}")

    # Sauvegarde des classes
    with open(config.classes_file, 'w') as f:
        json.dump(class_names, f)

    # Optimisation (Cache & Prefetch)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, len(class_names)


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Sequential:
    """Construit et compile l'architecture du modèle CNN."""
    model = models.Sequential([

        layers.Input(shape=input_shape),
        layers.Rescaling(1. / 255),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_history(history, filename: str) -> None:
    """
    Affiche et sauvegarde les courbes de Loss et Accuracy.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Graphique Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # Graphique Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    # Sauvegarde
    plt.savefig(filename)
    logging.info(f"Graphique sauvegardé sous : {filename}")

    # Affichage (non bloquant si possible, ou à fermer pour continuer)
    try:
        plt.show()
    except Exception:
        pass  # Ignore si pas d'interface graphique disponible

def train_model(model: models.Model,
                train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                config: TrainConfig) -> None:
    """Lance l'entraînement avec les callbacks appropriés."""
    logging.info("\n--- Démarrage de l'entraînement ---")

    my_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=config.model_filename,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=my_callbacks
    )

    loss, accuracy = model.evaluate(val_ds, verbose=1)
    logging.info(f"Accuracy validée sur le set de validation : {accuracy * 100:.2f}%")

    # Sauvegarde finale si le checkpoint n'a pas été créé (cas rare)
    if not os.path.exists(config.model_filename):
        model.save(config.model_filename)

    plot_history(history, config.plot_filename)


def create_final_archive(config: TrainConfig) -> None:
    """Crée l'archive ZIP contenant le modèle et le dataset transformé."""
    logging.info("Création de l'archive ZIP finale...")

    try:
        with zipfile.ZipFile(config.zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Ajout du modèle
            if os.path.exists(config.model_filename):
                zipf.write(config.model_filename, arcname="best_leaf_model.keras")

            # Ajout du fichier de classes
            if os.path.exists(config.classes_file):
                zipf.write(config.classes_file)

            # Ajout récursif du dataset
            # On navigue dans masked_dataset
            for root, _, files in os.walk(config.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calcul du chemin relatif pour l'archive
                    # Ex: masked_dataset/Apple/img.jpg -> masked_dataset/Apple/img.jpg
                    arcname = os.path.relpath(file_path, start=config.output_dir.parent)
                    zipf.write(file_path, arcname)

        logging.info(f"Terminé ! Archive générée : {config.zip_filename}")

    except Exception as e:
        logging.error(f"Erreur lors de la création du ZIP : {e}")


def main() -> None:
    """Point d'entrée principal."""
    # 1. Validation des entrées
    source_dir = validate_arguments()
    config = TrainConfig()

    # 2. Préparation
    prepare_directories(config)

    # 3. Pipeline de transformation
    run_preprocessing_pipeline(source_dir, config)

    # 4. Chargement des données
    train_ds, val_ds, num_classes = create_datasets(config)

    # 5. Construction et Entraînement
    model = build_model(
        input_shape=(config.img_height, config.img_width, 3),
        num_classes=num_classes
    )

    train_model(model, train_ds, val_ds, config)

    # 6. Export
    create_final_archive(config)


if __name__ == "__main__":
    main()