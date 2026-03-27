# =========================================================
# 📁 BASE PATH (Docker Airflow)
# =========================================================
BASE_DIR = "/opt/airflow/data"

# =========================================================
# 📦 DATA SOURCES
# =========================================================
ZIP_DIR = f"{BASE_DIR}/data_zip"
EXTRACT_DIR = f"{BASE_DIR}/dataset_extrait"
ANNOTATION_ZIP_PATH = f"{ZIP_DIR}/Annotations.zip"
ANNOTATION_EXTRACT_DIR = f"{BASE_DIR}/dataset_raw_annotation"

# =========================================================
# 📊 DATASETS
# =========================================================
FINAL_DATASET = f"{BASE_DIR}/dataset_type_stages"

DATASET_MATURITY_DIR = f"{BASE_DIR}/dataset_classification/dataset_maturity"
DATASET_VARIETY_DIR = f"{BASE_DIR}/dataset_classification/dataset_variety"

# =========================================================
# 🎯 DETECTION DATASET
# =========================================================
DATASET_DETECTION_DIR = f"{BASE_DIR}/dataset_detection"

TARGET_IMAGE_DIR = f"{DATASET_DETECTION_DIR}/data_merged/images"
TARGET_LABEL_DIR = f"{DATASET_DETECTION_DIR}/data_merged/labels"

SPLIT_DIR = f"{DATASET_DETECTION_DIR}/data_splited"

# =========================================================
# 🤖 MODELS
# =========================================================
MODEL_OUTPUT_DIR = "/opt/airflow/models"
YOLO_RUNS_DIR = "/opt/airflow/analyse_configuration/detection/runs"

# =========================================================
# 🔁 MERGE RULES
# =========================================================
MERGE_RULES = {
    "Boufagous": ["Boufagous", "Boufagous2", "Boufagous3"],
    "Boumajhoul": ["Boumajhoul", "Boumajhoul2"],
    "kholt": ["kholt"],
    "bouisthami": ["bouisthami"]
}