# classification_config.py
"""
Configuration pour les pipelines de classification
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analyse_configuration"

# Configuration pour la classification par variété
VARIETY_CONFIG = {
    "name": "variety",
    "experiment_name": "Variety_Classification",
    "dataset_dir": str(DATA_DIR / "dataset_classification" / "dataset_variety"),
    "output_dir": str(DATA_DIR / "dataset_classification" / "data_splited_v"),
    "model_path": str(MODELS_DIR / "model_classes_variete_googlenet_model.pth"),
    "analysis_dir": str(ANALYSIS_DIR / "variete_analyse"),
    "num_classes": 4,
    "class_names": ['Boufagous', 'bouisthami', 'Boumajhoul', 'kholt'],
    "image_size": 224,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.0001,
    "target_per_class": 1500,
    # "device": "cuda",
    "device": "cpu"
}

# Configuration pour la classification par maturité
MATURITY_CONFIG = {
    "name": "maturity",
    "experiment_name": "Maturity_Classification",
    "dataset_dir": str(DATA_DIR / "dataset_classification" / "dataset_maturity"),
    "output_dir": str(DATA_DIR / "dataset_classification" / "data_splited_m"),
    "model_path": str(MODELS_DIR / "model_classes_maturity_googlenet_model.pth"),
    "analysis_dir": str(ANALYSIS_DIR / "maturity_analyse"),
    "num_classes": 4,
    "class_names": ['S1', 'S2', 'S3', 'S4'],
    "image_size": 224,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.0001,
    "target_per_class": 1500,
    # "device": "cuda",
    "device": "cpu"
}