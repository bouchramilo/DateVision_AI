from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from typing import Dict, Any
import base64
import cv2
import numpy as np
import io
from PIL import Image
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.repositories.history_repository import save_full_result
from app.repositories.prediction_repository import run_prediction_pipeline
from app.schemas.user_schema import User
from app.utils.image_util import image_to_base64

pridect_router = APIRouter(prefix="/pridect", tags=["pridection"])


# =========================================================
#  ENDPOINT : PREDICT
# =========================================================
@pridect_router.post("/predict", response_model=Dict[str, Any])
async def predict(
    db: Session = Depends(get_db), 
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)):
    """
    Upload image → Predict → Return detections + annotated image
    """

    #  Validation simple
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        #  Lire image
        image_bytes = await file.read()
        
        #  Prediction via Repository
        result = run_prediction_pipeline(io.BytesIO(image_bytes))

        #  Convertir image annotée → base64
        img_base64 = image_to_base64(result["annotated_image"])
        
        save_full_result(db=db, user_id=current_user.id, image=img_base64, result=result)

        return {
            "success": True,
            "detections": result["detections"],
            "report": result["report"],
            "image": img_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))