# Local imports
from app.services.detection_service import get_yolo_model
from app.services.maturity_service import get_maturity_model
from app.services.variety_service import get_variety_model

# =========================================================
# 🔹 GLOBAL MODEL INITIALIZATION
# =========================================================

def load_all_models():
    """
    Initialise tous les modèles AI (YOLO, Maturité, Variété) au démarrage 
    pour éviter les lenteurs lors du premier appel API.
    """
    print("🚀 Chargement des modèles AI de production...")

    # 1. Détection YOLO
    get_yolo_model()
    print("✅ YOLOv8 detection model loaded")

    # 2. Maturité classifier
    get_maturity_model()
    print("✅ Maturity classifier loaded")

    # 3. Variété classifier
    get_variety_model()
    print("✅ Variety classifier loaded")

    print("🎉 Tous les modèles sont chargés et prêts !")