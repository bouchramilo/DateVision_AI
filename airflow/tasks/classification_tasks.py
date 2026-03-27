# airflow/tasks/classification_tasks.py
"""
Tâches Airflow pour la classification
"""

import os
import pandas as pd
from airflow.decorators import task
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.pipeline_classification import *


# ============================================================================
# tack : load and explore data
# ============================================================================
@task
def load_and_explore_data(config):
    """Tâche : Chargement et exploration des données"""
    print(f"📊 Chargement du dataset: {config['dataset_dir']}")
    
    # Charger les images
    image_paths, labels, total_images, problem_files = load_dataset_images(
        str(config['dataset_dir'])
    )
    
    # Statistiques
    class_counts = Counter(labels)
    
    return {
        "total_images": total_images,
        "valid_images": len(image_paths),
        "problem_files": problem_files,
        "class_counts": dict(class_counts),
        "num_classes": len(class_counts)
    }


# ============================================================================
# tack : split dataset 
# ============================================================================
@task
def split_dataset_task(config, exploration_results):
    """Tâche : Split du dataset en train/val/test"""
    print(f"📂 Split du dataset: {config['output_dir']}")
    
    output_dir = split_dataset(
        str(config['dataset_dir']),
        str(config['output_dir']),
        ratios=(0.7, 0.2, 0.1),
        seed=42
    )
    
    # Vérifier les splits
    stats = compute_dataset_stats(str(config['output_dir']))
    
    return {
        "output_dir": output_dir,
        "split_stats": stats.to_dict('records')
    }


# ============================================================================
# tack : Compute dataset stats
# ============================================================================
def compute_dataset_stats(dataset_dir, extensions=(".jpg", ".jpeg", ".png")):
    """Calcule les statistiques des splits"""
    stats = []
    
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path):
            continue
        
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            n_images = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(extensions)])
            
            stats.append({
                "Split": split,
                "Classe": class_name,
                "Nombre_images": n_images
            })
    
    return pd.DataFrame(stats)


# ============================================================================
# tack : prepare dataset
# ============================================================================
@task
def prepare_datasets(config):
    """Tâche : Préparation des datasets avec transformations"""
    
    # Chemins des splits
    train_dir = Path(config['output_dir']) / "train"
    val_dir = Path(config['output_dir']) / "val"
    test_dir = Path(config['output_dir']) / "test"
    
    # Transformations
    train_transforms = get_transforms(config['image_size'], augment=True)
    val_test_transforms = get_transforms(config['image_size'], augment=False)
    
    # Créer les datasets
    train_dataset = ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = ImageFolder(str(val_dir), transform=val_test_transforms)
    test_dataset = ImageFolder(str(test_dir), transform=val_test_transforms)
    
    return {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "class_names": train_dataset.classes,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir)
    }


# ============================================================================
# tack : create balanced loaders
# ============================================================================
@task
def create_balanced_loaders(config, dataset_info):
    """Tâche : Création des loaders équilibrés"""
    
    # Recharger les datasets avec transformations
    train_dir = dataset_info['train_dir']
    val_dir = dataset_info['val_dir']
    test_dir = dataset_info['test_dir']
    
    # Transformations
    train_transforms = get_transforms(config['image_size'], augment=False)
    val_test_transforms = get_transforms(config['image_size'], augment=False)
    augment_transforms = get_transforms(config['image_size'], augment=True)
    
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    
    # Créer le sampler équilibré
    sampler = create_balanced_sampler(
        train_dataset, 
        target_per_class=config['target_per_class'],
        augment_transforms=augment_transforms
    )
    
    # Créer les loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=sampler
    )
    
    val_dataset = ImageFolder(val_dir, transform=val_test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    test_dataset = ImageFolder(test_dir, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return {
        "batch_size": config['batch_size'],
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader)
    }


# ============================================================================
# tack : train model 
# ============================================================================
@task
def train_model_task(config, dataset_info, loaders_info):

    from pathlib import Path

    # ✅ Normalisation
    model_path_obj = Path(config['model_path'])

    train_dir = dataset_info['train_dir']
    val_dir = dataset_info['val_dir']

    train_transforms = get_transforms(config['image_size'], augment=True)
    val_transforms = get_transforms(config['image_size'], augment=False)

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    sampler = create_balanced_sampler(
        train_dataset,
        target_per_class=config['target_per_class']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device(config['device'])
    model = create_model(config['num_classes'], device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config['learning_rate'])
    loss_function = torch.nn.CrossEntropyLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_funct=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=config['epochs'],
        experiment_name=config.get('experiment_name', 'Default_Classification')
    )

    model_path = save_model(
        model,
        optimizer,
        history,
        str(model_path_obj.parent),
        model_path_obj.name,
        mlflow_run_id=history.get("mlflow_run_id")
    )

    return {
        "model_path": str(model_path),
        "history": history,
        "final_train_acc": history['train_accuracy'][-1],
        "final_val_acc": history['val_accuracy'][-1]
    }

# ============================================================================
# tack : evaluate model 
# ============================================================================
@task
def evaluate_model_task(config, dataset_info, training_results):
    """Tâche : Évaluation du modèle sur le test set"""
    
    # Recharger le modèle
    test_dir = dataset_info['test_dir']
    test_transforms = get_transforms(config['image_size'], augment=False)
    test_dataset = ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    device = torch.device(config['device'])
    model = create_model(config['num_classes'], device)
    
    # Charger les poids
    checkpoint = torch.load(str(training_results['model_path']), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Évaluation
    loss_function = torch.nn.CrossEntropyLoss()
    evaluation = evaluate_model(
        model, 
        test_loader, 
        loss_function, 
        device, 
        config['class_names']
    )
    
    # Créer les visualisations
    os.makedirs(config['analysis_dir'], exist_ok=True)
    
    # Matrice de confusion
    plot_confusion_matrix(
        evaluation['confusion_matrix'],
        config['class_names'],
        save_path=config['analysis_dir']
    )
    
    # Courbes d'entraînement
    plot_training_history(
        training_results['history'],
        config['epochs'],
        save_path=config['analysis_dir']
    )
    
    return {
        "test_accuracy": evaluation['accuracy'],
        "test_loss": evaluation['loss'],
        "classification_report": evaluation['classification_report'],
        "confusion_matrix": evaluation['confusion_matrix'].tolist(),
        "analysis_dir": str(config['analysis_dir'])
    }