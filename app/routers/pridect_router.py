from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import base64
import cv2
import numpy as np
import io
from PIL import Image

from app.services.predict_variety import predict_image


pridect_router = APIRouter(prefix="/pridect", tags=["pridection"])


# =========================================================
# 🔹 UTILS
# =========================================================
def image_to_base64(img: np.ndarray) -> str:
    """Convertit image numpy → base64"""
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Error encoding image")
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================
# 🔹 ENDPOINT : PREDICT
# =========================================================
@pridect_router.post("/predict", response_model=Dict[str, Any])
async def predict(file: UploadFile = File(...)):
    """
    Upload image → Predict → Return detections + annotated image
    """

    # 🔹 Validation simple
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # 🔹 Lire image (file-like)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # 🔹 Prediction
        result = predict_image(image)

        # 🔹 Convertir image annotée → base64
        img_base64 = image_to_base64(result["annotated_image"])

        return {
            "success": True,
            "detections": result["detections"],
            "image": img_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))