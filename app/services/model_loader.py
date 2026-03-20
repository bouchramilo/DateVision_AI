from app.services.detection_service import get_yolo_model
from app.services.maturity_service import get_maturity_model
from app.services.variety_service import get_variety_model

# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def load_all_models():
    print("🚀 Chargement des modèles...")

    get_yolo_model()
    print("✅ YOLO loaded")

    get_maturity_model()
    print("✅ Maturity model loaded")

    get_variety_model()
    print("✅ Variety model loaded")

    print("🎉 Tous les modèles sont prêts !")