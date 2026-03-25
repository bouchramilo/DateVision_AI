from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms

# =========================================================
# LOAD IMAGE
# =========================================================
def load_image(image_file) -> Image.Image:
    """Charge une image (file, buffer ou PIL)"""
    if isinstance(image_file, Image.Image):
        return image_file.convert("RGB")
    return Image.open(image_file).convert("RGB")


# =========================================================
# CONVERSIONS
# =========================================================
def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    if pil_img is None:
        raise TypeError("pil_img cannot be None")
    return np.array(pil_img)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# =========================================================
# DETECTION PREPROCESS (YOLO)
# =========================================================
def preprocess_for_detection(image_file) -> np.ndarray:
    """
    file → PIL → numpy → BGR
    """
    pil_img = load_image(image_file)
    img = pil_to_numpy(pil_img)
    img_bgr = rgb_to_bgr(img)
    return img_bgr


# =========================================================
# CLASSIFICATION PREPROCESS (Torch)
# =========================================================
def get_classification_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# =========================================================
# CLASSIFICATION PREPROCESS 
# =========================================================
def preprocess_for_classification(image, device: torch.device):
    """
    PIL → Tensor
    """
    if not isinstance(image, Image.Image):
        image = load_image(image)

    transform = get_classification_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    return tensor