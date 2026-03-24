from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from tasks.etl_functions import *
# ===================================================================
BASE_DIR = "/opt/airflow/data"

ZIP_DIR = f"{BASE_DIR}/data_zip"
EXTRACT_DIR = f"{BASE_DIR}/dataset_extrait"
FINAL_DATASET = f"{BASE_DIR}/dataset_type_stages"
DATASET_MATURITY_DIR = f"{BASE_DIR}/dataset_classification/dataset_maturity"
DATASET_VARIETY_DIR = f"{BASE_DIR}/dataset_classification/dataset_variety"
ANNOTATION_ZIP_PATH = f"{BASE_DIR}/data_zip/Annotations.zip"
ANNOTATION_EXTRACT_DIR = f"{BASE_DIR}/dataset_raw_annotation"
TARGET_LABEL_DIR = f"{BASE_DIR}/dataset_detection/data_merged/labels"
TARGET_IMAGE_DIR = f"{BASE_DIR}/dataset_detection/data_merged/images"
MERGE_RULES = {
    "Boufagous": ["Boufagous", "Boufagous2", "Boufagous3"],
    "Boumajhoul": ["Boumajhoul", "Boumajhoul2"],
    "kholt": ["kholt"],
    "bouisthami": ["bouisthami"]
}


# ============================================================================
# DAG : Extract -> Transform -> Load
# ============================================================================
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
}

dag = DAG(
    'etl_dates_pipeline',
    default_args=default_args,
    description='ETL pipeline pour détection et classification des dattes',
    schedule_interval='@yearly',
    catchup=False
)
# ===================================================================
# --------------------------
# Fonctions pour PythonOperator
# --------------------------
def task_extract_zips():
    extract_files_zip(zip_directory = ZIP_DIR, extract_directory = EXTRACT_DIR)

def debug_extraction():
    base = "/opt/airflow/data/dataset_extrait"
    for root, dirs, files in os.walk(base):
        print("📁", root)
        for d in dirs:
            print("   └──", d)
            
def task_merge_dataset():
    merge_dataset_by_stage(FINAL_DATASET, MERGE_RULES)

def task_build_variety():
    build_variety_dataset(FINAL_DATASET, DATASET_VARIETY_DIR)

def task_build_maturity():
    build_maturity_dataset(FINAL_DATASET, DATASET_MATURITY_DIR)

def task_extract_annotations():
    extract_annotations(ANNOTATION_ZIP_PATH, ANNOTATION_EXTRACT_DIR)

def task_copy_annotations():
    copy_annotations(
        os.path.join(ANNOTATION_EXTRACT_DIR, "Annotations/object detection/date fruits detection"),
        TARGET_LABEL_DIR,
        ANNOTATION_EXTRACT_DIR
    )

def task_merge_images():
    merge_images(FINAL_DATASET, TARGET_IMAGE_DIR)

# ===================================================================
# --------------------------
# Tâches Airflow
# --------------------------
t1_extract_zips = PythonOperator(task_id='extract_zips', python_callable=task_extract_zips, dag=dag)
t1_0_debug_extraction = PythonOperator(task_id='debug_extraction', python_callable=debug_extraction, dag=dag)
t2_merge_dataset = PythonOperator(task_id='merge_dataset', python_callable=task_merge_dataset, dag=dag)
t3_build_variety = PythonOperator(task_id='build_variety_dataset', python_callable=task_build_variety, dag=dag)
t4_build_maturity = PythonOperator(task_id='build_maturity_dataset', python_callable=task_build_maturity, dag=dag)
t5_extract_annotations = PythonOperator(task_id='extract_annotations', python_callable=task_extract_annotations, dag=dag)
t6_copy_annotations = PythonOperator(task_id='copy_annotations', python_callable=task_copy_annotations, dag=dag)
t7_merge_images = PythonOperator(task_id='merge_images', python_callable=task_merge_images, dag=dag)

# ===================================================================
# --------------------------
# Dépendances
# --------------------------
t1_extract_zips >> t2_merge_dataset
t2_merge_dataset >> [t3_build_variety, t4_build_maturity] >> t7_merge_images
t5_extract_annotations >> t6_copy_annotations >> t7_merge_images