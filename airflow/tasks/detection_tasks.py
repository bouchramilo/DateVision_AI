from functions.pipeline_detection import *

# =========================================================
def task_clean(**kwargs):
    remove_images_without_labels(
        kwargs["images_dir"],
        kwargs["labels_dir"],
        kwargs["extensions"]
    )


# =========================================================
def task_split(**kwargs):
    split_dataset(
        kwargs["images_dir"],
        kwargs["labels_dir"],
        kwargs["base_dir"],
        kwargs["extensions"]
    )


# =========================================================
def task_train(**kwargs):
    # Retourner le dict pour XCom (qui contient mlflow_run_id)
    return train_model(
        kwargs["data_yaml"],
        kwargs["output_dir"]
    )


# =========================================================
def task_evaluate(**kwargs):
    evaluate_model(
        kwargs["model_path"],
        kwargs["data_yaml"]
    )


# =========================================================
def task_predict(**kwargs):
    predict_sample(
        kwargs["model_path"],
        kwargs["image_dir"]
    )


# =========================================================
def task_save(**kwargs):
    # Récupérer l'ID de run MLflow de la tâche d'entraînement via XCom
    ti = kwargs.get('ti')
    mlflow_run_id = None
    if ti:
        train_results = ti.xcom_pull(task_ids='train_model')
        if train_results and isinstance(train_results, dict):
            mlflow_run_id = train_results.get('mlflow_run_id')
            print(f"🔗 Récupération Run ID MLflow via XCom: {mlflow_run_id}")

    save_model(
        kwargs["model_path"],
        kwargs["output_path"],
        mlflow_run_id=mlflow_run_id
    )