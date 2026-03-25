from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from tasks.detection_tasks import *
from config.detection_config import *

# =========================================================================
# DAG : detection
# =========================================================================
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1), 
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="detection_pipeline",
    catchup=False,
    schedule=None,
    default_args=default_args,
    description="Pipeline YOLOv8 detection dattes",
    tags=["AI", "YOLO", "detection"]
) as dag:


# tasks --------------------------------
    clean = PythonOperator(
        task_id="clean_dataset",
        python_callable=task_clean,
        op_kwargs={
            "images_dir": TARGET_IMAGE_DIR,
            "labels_dir": TARGET_LABEL_DIR,
            "extensions": [".jpg", ".png", ".jpeg"]
        }
    )

    split = PythonOperator(
        task_id="split_dataset",
        python_callable=task_split,
        op_kwargs={
            "images_dir": TARGET_IMAGE_DIR,
            "labels_dir": TARGET_LABEL_DIR,
            "base_dir": DATASET_DETECTION_DIR,
            "extensions": [".jpg", ".png", ".jpeg"]
        }
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
        op_kwargs={
            "data_yaml": f"{DATASET_DETECTION_DIR}/data.yaml",
            "output_dir": YOLO_RUNS_DIR
        }
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate,
        op_kwargs={
            "model_path": f"{YOLO_RUNS_DIR}/train/weights/best.pt",
            "data_yaml": f"{DATASET_DETECTION_DIR}/data.yaml"
        }
    )

    predict = PythonOperator(
        task_id="predict_sample",
        python_callable=task_predict,
        op_kwargs={
            "model_path": f"{YOLO_RUNS_DIR}/train/weights/best.pt",
            "image_dir": f"{SPLIT_DIR}/test/images"
        }
    )

    save = PythonOperator(
        task_id="save_model",
        python_callable=task_save,
        op_kwargs={
            "model_path": f"{YOLO_RUNS_DIR}/train/weights/best.pt",
            "output_path": f"{MODEL_OUTPUT_DIR}/date_detector_model.pt"
        }
    )

# ordre d'execution des tasks --------------------------------------------
    clean >> split >> train >> evaluate >> predict >> save