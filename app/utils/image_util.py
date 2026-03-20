import base64
import cv2
import numpy as np


# =========================================================
# 🔹 UTILS
# =========================================================
def image_to_base64(img: np.ndarray) -> str:
    """Convertit image numpy → base64"""
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Error encoding image")
    return base64.b64encode(buffer).decode("utf-8")