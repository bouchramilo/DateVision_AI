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
    train_model(
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
    save_model(
        kwargs["model_path"],
        kwargs["output_path"]
    )