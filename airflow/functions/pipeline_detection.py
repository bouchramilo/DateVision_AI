import os
import random
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================
# CLEAN DATASET
# =========================================================
def remove_images_without_labels(images_dir, labels_dir, extensions):
    missing = []

    for img in os.listdir(images_dir):
        if os.path.splitext(img)[1].lower() in extensions:
            name = os.path.splitext(img)[0]
            if not os.path.exists(os.path.join(labels_dir, name + ".txt")):
                missing.append(img)

    for img in missing:
        os.remove(os.path.join(images_dir, img))

    print(f"🗑️ {len(missing)} images supprimées")


# =========================================================
# SPLIT DATASET
# =========================================================
def split_dataset(images_dir, labels_dir, base_dir, extensions,
                  train_ratio=0.7, val_ratio=0.2):

    splits = ["train", "val", "test"]

    for s in splits:
        os.makedirs(f"{base_dir}/data_splited/{s}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/data_splited/{s}/labels", exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith(tuple(extensions))]
    random.shuffle(images)

    n = len(images)
    train = images[:int(n*train_ratio)]
    val = images[int(n*train_ratio):int(n*(train_ratio+val_ratio))]
    test = images[int(n*(train_ratio+val_ratio)):]

    def move(files, split):
        for img in files:
            name = os.path.splitext(img)[0]

            shutil.move(
                os.path.join(images_dir, img),
                f"{base_dir}/data_splited/{split}/images/{img}"
            )

            lbl = os.path.join(labels_dir, name + ".txt")
            if os.path.exists(lbl):
                shutil.move(
                    lbl,
                    f"{base_dir}/data_splited/{split}/labels/{name}.txt"
                )

    move(train, "train")
    move(val, "val")
    move(test, "test")

    print("✅ Split terminé")


# =========================================================
# TRAIN
# =========================================================
def train_model(data_yaml, output_dir):

    os.environ["MLFLOW_DISABLED"] = "true"

    model = YOLO("yolov8n.pt")

    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=50,
        batch=16,
        project=output_dir,
        name="train",
        exist_ok=True
    )


# =========================================================
# EVALUATE
# =========================================================
def evaluate_model(model_path, data_yaml):

    model = YOLO(model_path)

    metrics = model.val(data=data_yaml)

    print("mAP50:", metrics.box.map50)


# =========================================================
# PREDICT
# =========================================================
def predict_sample(model_path, image_dir):

    model = YOLO(model_path)

    img = random.choice(os.listdir(image_dir))

    model.predict(
        source=os.path.join(image_dir, img),
        conf=0.25
    )


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model_path, output_path):

    model = YOLO(model_path)
    model.save(output_path)

    print("✅ modèle sauvegardé")