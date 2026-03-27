import os
import time
from ultralytics import YOLO
from PIL import Image
from typing import List, Dict, Any
import torch
from app.core.config import settings

from app.core.metrics import (
    YOLO_DETECTIONS_TOTAL,
    YOLO_OBJECTS_DETECTED,
    YOLO_INFERENCE_TIME,
    YOLO_CONFIDENCE_SCORE,
)

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

    # ── YOLO inference timing ─────────────────────────────────────────────
    start = time.perf_counter()
    try:
        results = model(image, device=DEVICE)[0]
    except Exception as exc:
        raise exc
    yolo_duration = time.perf_counter() - start
    YOLO_INFERENCE_TIME.observe(yolo_duration)
    # ─────────────────────────────────────────────────────────────────────

    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = (
            model.names.get(class_id, str(class_id))
            if hasattr(model, "names")
            else str(class_id)
        )

        detections.append({
            "bbox":     box.xyxy[0].tolist(),
            "score":    confidence,
            "class_id": class_id,
        })

        # ── YOLO metrics ──────────────────────────────────────────────────
        YOLO_DETECTIONS_TOTAL.labels(class_name=class_name).inc()
        YOLO_OBJECTS_DETECTED.labels(class_name=class_name).inc()
        YOLO_CONFIDENCE_SCORE.labels(class_name=class_name).set(confidence)
        # ─────────────────────────────────────────────────────────────────

    return detections