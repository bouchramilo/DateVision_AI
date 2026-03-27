# airflow/functions/pipeline_detection.py
import os
import random
import shutil
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS['mlflow'] = False
os.environ["ULTRALYTICS_MLFLOW"] = "false"
os.environ["ULTRALYTICS_WANDB"] = "false"
os.environ["ULTRALYTICS_TENSORBOARD"] = "false"

try:
    import sys
    sys.path.append('/opt/airflow')
    from functions.mlflow_utils import safe_mlflow_run, initialize_mlflow, get_mlflow_tracking_uri
    print("✅ MLflow utils importé avec succès")
except ImportError as e:
    print(f"⚠️ Erreur d'import MLflow utils: {e}")
    from contextlib import contextmanager
    
    @contextmanager
    def safe_mlflow_run(exp, name):
        class DummyRun:
            class DummyInfo:
                run_id = None
            info = DummyInfo()
        yield DummyRun()
    
    def initialize_mlflow(*args, **kwargs):
        return False
    
    def get_mlflow_tracking_uri():
        return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Initialisation MLflow
MLFLOW_AVAILABLE = initialize_mlflow()

if not MLFLOW_AVAILABLE:
    print("⚠️ pipeline_detection ==> MLflow non disponible, continuation sans logging")
else:
    print(f"✅ pipeline_detection ==> MLflow disponible")


# =========================================================
# CLEAN DATASET
# =========================================================
def remove_images_without_labels(images_dir, labels_dir, extensions):
    missing = []

    for img in os.listdir(images_dir):
        if os.path.splitext(img)[1].lower() in extensions:
            name = os.path.splitext(img)[0]
            if not os.path.exists(os.path.join(labels_dir, name + ".txt")):
                missing.append(img)

    for img in missing:
        os.remove(os.path.join(images_dir, img))

    print(f"🗑️ {len(missing)} images supprimées")


# =========================================================
# SPLIT DATASET
# =========================================================
def split_dataset(images_dir, labels_dir, base_dir, extensions,
                  train_ratio=0.7, val_ratio=0.2):

    splits = ["train", "val", "test"]

    for s in splits:
        os.makedirs(f"{base_dir}/data_splited/{s}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/data_splited/{s}/labels", exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith(tuple(extensions))]
    random.shuffle(images)

    n = len(images)
    train = images[:int(n*train_ratio)]
    val = images[int(n*train_ratio):int(n*(train_ratio+val_ratio))]
    test = images[int(n*(train_ratio+val_ratio)):]

    def move(files, split):
        for img in files:
            name = os.path.splitext(img)[0]

            shutil.move(
                os.path.join(images_dir, img),
                f"{base_dir}/data_splited/{split}/images/{img}"
            )

            lbl = os.path.join(labels_dir, name + ".txt")
            if os.path.exists(lbl):
                shutil.move(
                    lbl,
                    f"{base_dir}/data_splited/{split}/labels/{name}.txt"
                )

    move(train, "train")
    move(val, "val")
    move(test, "test")

    print("✅ Split terminé")


# =========================================================
# TRAIN
# =========================================================
def train_model(data_yaml, output_dir):
    """
    Entraîne le modèle YOLOv8
    """
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_yolov8_train"
    
    print(f"🚀 Début de l'entraînement YOLOv8")
    print(f"📁 Data YAML: {data_yaml}")
    print(f"📁 Output: {output_dir}")
    
    # Utiliser le context manager
    results = None
    try:
        with safe_mlflow_run("YOLOv8_detection", run_name) as run:
            import mlflow
            mlflow.set_tracking_uri(get_mlflow_tracking_uri())
            
            print(f"🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠📊 Logging dans MLflow - Run: {run} 🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠🟠")
            if run and run.info and run.info.run_id:
                print(f"📊 Logging dans MLflow - Run ID: {run.info.run_id}")
                import mlflow
                mlflow.log_params({
                    "epochs": 1,
                    "batch_size": 16,
                    "imgsz": 640,
                    "data_yaml": data_yaml,
                    "model": "yolov8n.pt"
                })
                mlflow.set_tags({
                    "model_type": "YOLOv8n",
                    "stage": "training",
                    "dataset_version": "v1.2",
                    "author": "Pipeline_Airflow"
                })
            else:
                print("⚠️ Run MLflow non démarré, continuation sans logging")

            model = YOLO("yolov8n.pt")

            results = model.train(
                data=data_yaml,
                imgsz=640,
                epochs=1,
                batch=16,
                project=output_dir,
                name="train",
                exist_ok=True,
                verbose=True
            )
            
            if run and run.info and run.info.run_id:
                import mlflow
                train_dir = os.path.join(output_dir, "train")
                if os.path.exists(train_dir):
                    try:
                        # Lire et logger les métriques
                        results_csv = os.path.join(train_dir, "results.csv")
                        if os.path.exists(results_csv):
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            if len(df) > 0:
                                last_row = df.iloc[-1]
                                for col in df.columns:
                                    if 'mAP' in col or 'loss' in col:
                                        try:
                                            value = float(last_row[col])
                                            mlflow.log_metric(col, value)
                                        except:
                                            pass
                    except Exception as e:
                        print(f"⚠️ Erreur lors du logging des métriques: {e}")
                    
                    mlflow.log_artifacts(train_dir, artifact_path="training_results")
                    print(f"✅ Résultats sauvegardés dans MLflow")
            
            print("✅ Entraînement terminé")
            
            return {
                "results": results,
                "mlflow_run_id": run.info.run_id if run else None,
                "train_dir": os.path.join(output_dir, "train")
            }
            
    except Exception as e:
        print(f"❌ Erreur dans l'entraînement: {e}")
        raise


# =========================================================
# EVALUATE
# =========================================================
def evaluate_model(model_path, data_yaml):
    """
    Évalue le modèle entraîné
    """
    print(f"🔍 Évaluation du modèle: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"📊 mAP50: {metrics.box.map50:.4f}")
    print(f"📊 mAP50-95: {metrics.box.map:.4f}")
    return metrics


# =========================================================
# PREDICT
# =========================================================
def predict_sample(model_path, image_dir):
    """
    Fait une prédiction sur un échantillon
    """
    model = YOLO(model_path)
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if images:
        img = random.choice(images)
        print(f"🔮 Prédiction sur: {img}")
        results = model.predict(
            source=os.path.join(image_dir, img),
            conf=0.25,
            save=True
        )
        print("✅ Prédiction terminée")
    else:
        print("⚠️ Aucune image trouvée pour la prédiction")


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model_path, output_path, mlflow_run_id=None):
    """
    Sauvegarde le modèle avec versionnement et log dans MLflow
    """
    import shutil
    import os
    from datetime import datetime
    import mlflow

    parent_dir = os.path.dirname(output_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"📁 Répertoire créé: {parent_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, extension = os.path.splitext(output_path)
    versioned_path = f"{base_name}_{timestamp}{extension}"

    try:
        
        shutil.copyfile(model_path, output_path)
        shutil.copyfile(model_path, versioned_path)
        
        print(f"✅ Modèle sauvegardé (PROD): {output_path}")
        print(f"✅ Modèle sauvegardé (VERSION): {versioned_path}")

        def _log_to_mlflow():
            if mlflow.active_run():
                mlflow.log_artifact(output_path, artifact_path="models")
                print(f"📦 Modèle loggué dans MLflow (Run actif)")
            elif mlflow_run_id:
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_artifact(output_path, artifact_path="models")
                    print(f"📦 Modèle loggué dans MLflow (Run ID: {mlflow_run_id})")

        try:
            _log_to_mlflow()
        except Exception as mlflow_e:
            print(f"⚠️ Erreur lors du log MLflow: {mlflow_e}")
        
        return output_path

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return None