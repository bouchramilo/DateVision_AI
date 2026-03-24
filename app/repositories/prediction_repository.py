import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import io

from app.services.detection_service import detect_objects
from app.services.variety_service import predict_variety
from app.services.maturity_service import predict_maturity
from app.services.upload_img_service import load_image, pil_to_numpy
from app.services.llm_service import generate_report
from app.utils.image_util import image_to_base64
# Labels

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
VARIETY_LABELS = ['Boufagous', 'bouisthami', 'Boumajhoul', 'kholt']
MATURITY_LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def run_prediction_pipeline(image_file) -> Dict[str, Any]:
    # 1. Load image
    pil_img = load_image(image_file)
    
    if pil_img is None:
        raise ValueError("Impossible de charger l'image")
    img_np = pil_to_numpy(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 2. Detect objects
    detections = detect_objects(pil_img)
    
    print("Detections:", detections)
    
    results = []
    annotated_img = img_bgr.copy()
    
    for det in detections:
        bbox = det["bbox"]
        score_det = det["score"]
        
        x1, y1, x2, y2 = map(int, bbox)

        h, w, _ = img_bgr.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop_bgr = img_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # Classification
        var_res = predict_variety(crop_pil)
        mat_res = predict_maturity(crop_pil)
        
        if var_res["class_id"] >= len(VARIETY_LABELS):
            var_res["class_id"] = 0  # fallback
        if mat_res["class_id"] >= len(MATURITY_LABELS):
            mat_res["class_id"] = 0
            
        var_class = VARIETY_LABELS[var_res["class_id"]]
        mat_class = MATURITY_LABELS[mat_res["class_id"]]
        
            
        res_item = {
            "bbox": bbox,
            "detection_score": score_det,
            "variety": var_class,
            "variety_score": var_res["confidence"],
            "maturity": mat_class,
            "maturity_score": mat_res["confidence"]
        }
        
        results.append(res_item)
        
        # Annotation
        label_text = f"{var_class} ({var_res['confidence']:.2f}) | {mat_class} ({mat_res['confidence']:.2f})"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    # Génération du rapport 
    report = generate_report(results)

    return {
        "detections": results,
        "report": report,
        "annotated_image": annotated_img
    }