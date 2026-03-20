import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from typing import Dict, Any
from app.services.upload_img_service import preprocess_for_classification
from app.core.config import settings

# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", settings.MODEL_CLASSIFICATION_BY_MATURITY)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

CLASSES = ['S1', 'S2', 'S3', 'S4']

# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_maturity_model(model_path: str = MODEL_PATH):
    global _model

    if _model is None:
        print("🚀 Chargement modèle maturité...")

        checkpoint = torch.load(model_path, map_location=DEVICE)

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model = models.googlenet(weights=None, aux_logits=True)

        model.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

        model.load_state_dict(state_dict, strict=False)

        model.to(DEVICE)
        model.eval()

        _model = model

    return _model


# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def predict_maturity(image):
    model = get_maturity_model()

    img_tensor = preprocess_for_classification(image, DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "class_id": int(pred.item()),
        "class_name": CLASSES[pred.item()],
        "confidence": float(conf.item())
    }