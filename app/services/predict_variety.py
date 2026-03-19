import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Any

from app.services.upload_img import load_image, pil_to_numpy  # utiliser tes fonctions existantes


# =========================================================
# 🔹 MODEL (Singleton)
# =========================================================
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_classes_variete_googlenet_model.pth")

_model = None

def get_model(model_path: str = DEFAULT_MODEL_PATH) -> torch.nn.Module:
    global _model
    if _model is None:
        # Charger le state_dict
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))["model_state_dict"]
        
        from torchvision.models import googlenet
        # Créer le modèle sans aux_logits car ils sont None dans le notebook
        model = googlenet(pretrained=False, aux_logits=False)
        
        # Redéfinir la couche 'fc' comme un Sequential block correspondant au notebook
        model.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 4)
        )
        
        model.load_state_dict(state_dict)
        model.eval()
        _model = model
    return _model


# =========================================================
# 🔹 PREPROCESS IMAGE
# =========================================================
def preprocess_image_torch(image_file) -> torch.Tensor:
    """
    Convertit PIL image → Tensor normalisé
    """
    pil_img = load_image(image_file)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # taille standard GoogLeNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(pil_img).unsqueeze(0)  # batch dimension
    return img_tensor


# =========================================================
# 🔹 PREDICTION
# =========================================================
def predict_image(image_file) -> Dict[str, Any]:
    """
    Pipeline complet pour GoogLeNet PyTorch :
    upload → preprocess → predict → postprocess
    """

    model = get_model()

    # 1️⃣ Preprocess
    img_tensor = preprocess_image_torch(image_file)

    # 2️⃣ Inference
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, class_idx = torch.max(probs, dim=1)

    class_idx = class_idx.item()
    conf = conf.item()

    # 3️⃣ Optionnel : annoter image (ici juste conversion en numpy)
    img_np = pil_to_numpy(load_image(image_file))

    return {
        "detections": [{
            "id": class_idx,
            "confidence": conf
        }],
        "annotated_image": img_np
    }