from PIL import Image
import numpy as np
import cv2


# =========================================================
# 🔹 LOAD IMAGE
# =========================================================
def load_image(image_file) -> Image.Image:
    """Charge une image depuis fichier, buffer ou déjà une PIL Image"""
    if isinstance(image_file, Image.Image):
        return image_file.convert("RGB")
    return Image.open(image_file).convert("RGB")


# =========================================================
# 🔹 CONVERSIONS
# =========================================================
def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# =========================================================
# 🔹 PREPROCESS PIPELINE
# =========================================================
def preprocess_image(image_file) -> np.ndarray:
    """
    Pipeline complet :
    file → PIL → numpy → BGR
    """
    pil_img = load_image(image_file)
    img = pil_to_numpy(pil_img)
    img_bgr = rgb_to_bgr(img)
    return img_bgr