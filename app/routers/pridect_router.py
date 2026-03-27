from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from typing import Dict, Any
import base64
import cv2
import numpy as np
import io
import time
from PIL import Image
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user
from app.repositories.history_repository import save_full_result
from app.repositories.prediction_repository import run_prediction_pipeline
from app.schemas.user_schema import User
from app.utils.image_util import image_to_base64

from app.core.metrics import (
    # Upload
    IMAGE_UPLOAD_TOTAL,
    IMAGE_UPLOAD_SIZE,
    # API
    API_REQUESTS_TOTAL,
    API_REQUEST_LATENCY,
    # API Errors
    API_ERRORS_TOTAL,
    # Business
    DATES_DETECTED_TOTAL,
    PREDICTIONS_PER_USER,
)

# =========================================================
# 🔹 CONFIGURATION
# =========================================================
pridect_router = APIRouter(prefix="/pridect", tags=["pridection"])


# =========================================================
# 🔹 ENDPOINTS
# =========================================================

@pridect_router.post("/predict", response_model=Dict[str, Any])
async def predict(
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)):
    """
    Lance le pipeline de prédiction sur une image uploadée.
    Retourne les détections, le rapport LLM et l'image annotée.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()

        # Metrics d'upload
        IMAGE_UPLOAD_TOTAL.inc()
        IMAGE_UPLOAD_SIZE.observe(len(image_bytes))

        pipeline_start = time.perf_counter()

        result = run_prediction_pipeline(io.BytesIO(image_bytes))

        latency = time.perf_counter() - pipeline_start
        API_REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        img_base64 = image_to_base64(result["annotated_image"])

        save_full_result(db=db, user_id=current_user.id, image=img_base64, result=result)

        PREDICTIONS_PER_USER.labels(user_id=str(current_user.id)).inc()
        API_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="200").inc()

        for det in result.get("detections", []):
            variety  = det.get("variety", "unknown")
            maturity = det.get("maturity", "unknown")

            DATES_DETECTED_TOTAL.labels(
                variety=variety,
                maturity=maturity,
            ).inc()

        return {
            "success": True,
            "detections": result["detections"],
            "report": result["report"],
            "image": img_base64,
            "processing_time": round(latency, 3)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Metrics d'erreur
        API_ERRORS_TOTAL.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        API_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))