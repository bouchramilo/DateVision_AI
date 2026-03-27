# airflow/functions/pipeline_classification.py
"""
Fonctions réutilisables pour les pipelines de classification
"""

import os
import random
import time
import numpy as np
import pandas as pd
import datetime
import mlflow
import mlflow.pytorch

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
try:
    from functions.mlflow_utils import safe_mlflow_run, initialize_mlflow
    if not initialize_mlflow(MLFLOW_TRACKING_URI):
        print("⚠️ pipeline_classification ==> MLflow non disponible, continuation sans logging")
except ImportError:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    from contextlib import contextmanager

    @contextmanager
    def safe_mlflow_run(exp, run_name):
        mlflow.set_experiment(exp)
        run = mlflow.start_run(run_name=run_name)
        try:
            yield run
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

    def initialize_mlflow(*args, **kwargs):
        return False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import cv2
import splitfolders

# Import conditionnel de torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, SubsetRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some functions will be disabled.")

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================================
# function : set seed
# ============================================================================
def set_seed(seed=42):
    """Fixer les seeds pour la reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# function : load dataset images
# ============================================================================
def load_dataset_images(dataset_dir, extensions=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    """
    Parcourt un répertoire de dataset pour extraire les chemins d'images et leurs labels.
    """
    image_paths = []
    labels = []
    problem_files = 0
    total_images = 0
    
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Le dossier {dataset_dir} n'existe pas")
    
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for file_name in os.listdir(class_path):
            total_images += 1
            file_path = os.path.join(class_path, file_name)
            
            if not file_name.lower().endswith(extensions):
                problem_files += 1
                continue
            
            try:
                img = cv2.imread(file_path)
                if img is None:
                    problem_files += 1
                    continue
                image_paths.append(file_path)
                labels.append(class_name)
            except Exception as e:
                problem_files += 1
                print(f"Erreur chargement: {file_path} -> {e}")
    
    return image_paths, labels, total_images, problem_files


# ============================================================================
# function : split dataset to train, val, test
# ============================================================================
def split_dataset(input_dir, output_dir, ratios=(0.7, 0.2, 0.1), seed=42):
    """
    Split le dataset en train/val/test
    """
    os.makedirs(output_dir, exist_ok=True)
    
    splitfolders.ratio(
        input_dir,
        output=output_dir,
        seed=seed,
        ratio=ratios,
        move=True
    )
    
    return output_dir


# ============================================================================
# function : get transforms
# ============================================================================
def get_transforms(image_size=224, augment=False):
    """
    Crée les transformations pour les images
    """
    if not TORCH_AVAILABLE:
        return None
    
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# ============================================================================
# function : create balanced sampler
# ============================================================================
def create_balanced_sampler(train_dataset, target_per_class=1500, augment_transforms=None):
    """
    Crée un sampler équilibré pour le dataset d'entraînement
    """
    if not TORCH_AVAILABLE:
        return None
    
    targets = np.array(train_dataset.targets)
    classes = np.unique(targets)
    
    balanced_indices = []
    
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        
        if len(cls_indices) > target_per_class:
            # Under-sampling
            sampled = np.random.choice(cls_indices, target_per_class, replace=False)
        else:
            # Over-sampling avec augmentation
            sampled = np.random.choice(cls_indices, target_per_class, replace=True)
            if augment_transforms:
                train_dataset.transform = augment_transforms
        
        balanced_indices.extend(sampled)
    
    np.random.shuffle(balanced_indices)
    return SubsetRandomSampler(balanced_indices)


# ============================================================================
# function : create model
# ============================================================================
def create_model(num_classes, device):
    """
    Crée et configure le modèle GoogLeNet
    """
    if not TORCH_AVAILABLE:
        return None
    
    model = models.googlenet(pretrained=True)
    
    # Modifier la couche FC
    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    # Déterminer le device dynamiquement
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device("cpu")
    
    model = model.to(device_obj)
    return model


# ============================================================================
# function : train model
# ============================================================================
def train_model(model, train_loader, val_loader, loss_funct, optimizer, device, epochs=40, experiment_name="classification_variety"):
    """
    Entraîne le modèle
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_googlenet_train"
    
    with safe_mlflow_run(experiment_name, run_name) as run:
        if run is not None:
            mlflow.log_params({
                "epochs": epochs,
                "optimizer": type(optimizer).__name__,
                "loss_function": type(loss_funct).__name__
            })
            mlflow.set_tags({
                "model_type": "GoogLeNet",
                "stage": "training",
                "dataset_version": "v1.2",
                "author": "Pipeline_Airflow"
            })
        else:
            print("⚠️ Run MLflow non démarré, continuation sans logging")
        
        train_loss = []
        train_accuracy = []
        val_accuracy = []
        
        print(f"▶️ Début de l'entraînement sur {device}...")
        start_time = time.time()
        
        for epoch in range(epochs):
            start_time_epoch = time.time()
            print(f"\n🔹 Epoch [{epoch+1}/{epochs}]")
            
            # Phase train
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_funct(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / total
            train_acc = correct / total
            
            # Phase validation
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()
            
            val_acc = correct_val / total_val
            
            # Sauvegarde
            train_loss.append(epoch_loss)
            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)
            
            if run is not None:
                mlflow.log_metrics({
                    "train_loss": epoch_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc
                }, step=epoch)
            
            print(f"    Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            end_time_epoch = time.time()
            training_time_epoch = end_time_epoch - start_time_epoch
            print(f"⏱️ Temps Epoch : {training_time_epoch/60:.2f} min")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n✅ Entraînement terminé ! Temps total : {training_time/60:.2f} minutes")
    
    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "training_time": training_time,
        # "mlflow_run_id": run.info.run_id
        "mlflow_run_id": run.info.run_id if run else None
    }


# ============================================================================
# function : evaluate model
# ============================================================================
def evaluate_model(model, data_loader, loss_function, device, class_names, mlflow_run_id=None):
    """
    Évalue le modèle et retourne les métriques
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Rapport de classification
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    if mlflow_run_id:
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metric("test_loss", avg_loss)
                mlflow.log_metric("test_accuracy", accuracy)
                
                # Log simplified classification report metrics safely
                for class_name in class_names:
                    if class_name in report:
                        mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                        mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                        mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])
        except Exception as e:
            print(f"⚠️ Run MLflow non disponible pour l'évaluation: {e}")
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": all_labels,
        "y_pred": all_preds
    }


# ============================================================================
# function : plot_training history
# ============================================================================
def plot_training_history(history, epochs, save_path=None):
    """
    Affiche et sauvegarde les courbes d'entraînement
    """
    train_loss = history['train_loss']
    train_accuracy = history['train_accuracy']
    val_accuracy = history['val_accuracy']
    
    epochs_range = list(range(1, epochs + 1))
    
    # Courbe du Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_loss, marker='o', label='Training Loss')
    plt.title("Évolution du Loss pendant l'entraînement")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/loss_curve.png")
    plt.close()
    
    # Courbe de l'Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_accuracy, marker='o', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, marker='s', label='Validation Accuracy')
    plt.title("Évolution de l'Accuracy pendant l'entraînement")
    plt.xlabel("Époques")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/accuracy_curve.png")
    plt.close()


# ============================================================================
# function : plot confusion matrix
# ============================================================================
def plot_confusion_matrix(cm, class_names, title="Matrice de confusion", save_path=None):
    """
    Affiche et sauvegarde la matrice de confusion
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        mat_path = f"{save_path}/confusion_matrix.png"
        plt.savefig(mat_path)
        try:
            if mlflow.active_run():
                mlflow.log_artifact(mat_path, artifact_path="evaluation_plots")
        except Exception:
            pass
    plt.close()


# ============================================================================
# function : save model
# ============================================================================
def save_model(model, optimizer, history, save_path, model_name, mlflow_run_id=None):
    """
    Sauvegarde le modèle entraîné (version PROD + version datée)
    """
    if not TORCH_AVAILABLE:
        return None
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Sauvegarde Production
    full_path = os.path.join(save_path, model_name)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }
    torch.save(save_dict, full_path)
    print(f"✅ Modèle sauvegardé (PROD) : {full_path}")

    # 2. Sauvegarde Versionnée
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    name_part, ext_part = os.path.splitext(model_name)
    versioned_name = f"{name_part}_{timestamp}{ext_part}"
    versioned_path = os.path.join(save_path, versioned_name)
    
    torch.save(save_dict, versioned_path)
    print(f"✅ Modèle sauvegardé (VERSION) : {versioned_path}")
    
    # 3. Log MLflow (Re-ouvrir le run si un ID est fourni)
    def _log_mlflow():
        if mlflow.active_run():
            mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=model_name.replace(".pth", ""))
            mlflow.log_artifact(full_path, artifact_path="saved_models")
            print(f"📦 Modèle loggué dans MLflow (Run actif)")
        elif mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=model_name.replace(".pth", ""))
                mlflow.log_artifact(full_path, artifact_path="saved_models")
                print(f"📦 Modèle loggué dans MLflow (Run ID: {mlflow_run_id})")

    try:
        _log_mlflow()
    except Exception as e:
        print(f"⚠️ Erreur lors du log du modèle MLflow: {e}")
    
    return full_path