import os
from ultralytics import YOLO
from PIL import Image
from typing import List, Dict, Any
import torch
from app.core.config import settings
# from app.services.upload_img_service import preprocess_for_detection

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", settings.MODEL_DETECTION)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None


# ==============================
# LOAD MODEL (Singleton)
# ==============================
def get_yolo_model(model_path: str = MODEL_PATH) -> YOLO:
    global _model

    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modèle YOLO introuvable : {model_path}")

        print("🚀 Chargement YOLO...")
        _model = YOLO(model_path)
        _model.to(DEVICE)

    return _model


# ==============================
# DETECTION
# ==============================
def detect_objects(image: Image.Image) -> List[Dict[str, Any]]:
    model = get_yolo_model()

    results = model(image, device=DEVICE)[0]

    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        detections.append({
            "bbox": box.xyxy[0].tolist(),
            "score": float(box.conf[0]),
            "class_id": int(box.cls[0])
        })

    return detections