import os
import zipfile
import shutil
from pathlib import Path
import time

EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
BASE_DIR = "/opt/airflow/data"
# --------------------------
# Extraction des fichiers ZIP
# --------------------------
def list_zip_files(zip_dir):
    return [f for f in os.listdir(zip_dir) if f.endswith(".zip")]

def count_images_in_zip(zip_path, extensions=EXTENSIONS):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
    return len([f for f in files if f.lower().endswith(extensions)])

# ===================================================================
def extract_files_zip(zip_directory:str, extract_directory: str):
    start_time_extract = time.time()
    for zip_file in os.listdir(zip_directory):
        if zip_file.endswith(".zip"):
            class_name = zip_file.replace(".zip", "")

            extract_path = os.path.join(extract_directory, class_name)
            os.makedirs(extract_path, exist_ok=True)

            zip_path = os.path.join(zip_directory, zip_file)
            print(f"📦 Extraction de {zip_file} → {extract_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    end_time_extract = time.time()
    print(f"    🕐 DUREE : {(end_time_extract-start_time_extract)/60:.2f} minutes")
    
# ===================================================================

def resolve_stage_path(base_dir, src_type, stage):
    """
    Gère les 2 cas :
    1. dataset_extrait/Boufagous/S1
    2. dataset_extrait/Boufagous/Boufagous/S1
    """

    # Cas simple
    path1 = os.path.join(base_dir, src_type, stage)

    # Cas imbriqué
    path2 = os.path.join(base_dir, src_type, src_type, stage)

    if os.path.isdir(path1):
        return path1
    elif os.path.isdir(path2):
        return path2
    else:
        return None
    


# --------------------------
# Fusion des datasets par variété / maturité
# --------------------------
def create_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_images(src_dir, dst_dir, extensions=EXTENSIONS, prefix=""):
    if not src_dir or not os.path.isdir(src_dir):
        print(f"⚠️ Dossier introuvable: {src_dir}")
        return 0

    count = 0

    for file in Path(src_dir).iterdir():
        if file.is_file() and file.suffix.lower() in [ext.lower() for ext in extensions]:
            dst_file = os.path.join(
                dst_dir,
                f"{prefix}_{file.name}" if prefix else file.name
            )
            shutil.copy(file, dst_file)
            count += 1

    return count

def merge_dataset_by_stage(final_dataset, merge_rules, stages=["S1", "S2", "S3", "S4"]):

    for final_class, sources in merge_rules.items():

        print(f"\n🌴 Fusion → {final_class}")

        final_class_dir = os.path.join(final_dataset, final_class)
        create_directory(final_class_dir)

        for stage in stages:

            final_stage_dir = os.path.join(final_class_dir, stage)
            create_directory(final_stage_dir)

            total_stage = 0

            for src_type in sources:

                stage_path = resolve_stage_path(
                    os.path.join(BASE_DIR, "dataset_extrait"),
                    src_type,
                    stage
                )

                total_stage += copy_images(stage_path, final_stage_dir)

            if total_stage > 0:
                print(f"   └── {stage} : {total_stage} images")

# --------------------------
# Préparer dataset maturité
# --------------------------
def build_maturity_dataset(source_dataset, target_dataset):

    create_directory(target_dataset)
    total_images = 0

    for variety in os.listdir(source_dataset):

        variety_path = os.path.join(source_dataset, variety)

        if not os.path.isdir(variety_path):
            continue

        for stage in os.listdir(variety_path):

            stage_path = os.path.join(variety_path, stage)

            if not os.path.isdir(stage_path):
                continue

            target_stage_dir = os.path.join(target_dataset, stage)
            create_directory(target_stage_dir)

            total_images += copy_images(stage_path, target_stage_dir, prefix=variety)

    print(f"\n✅ Total images maturité: {total_images}")
    return total_images

# --------------------------
# Préparer dataset variété
# --------------------------
def build_variety_dataset(final_dataset, dataset_variete_dir):

    create_directory(dataset_variete_dir)

    for variety in os.listdir(final_dataset):

        variety_path = os.path.join(final_dataset, variety)

        if not os.path.isdir(variety_path):
            continue

        target_variety_dir = os.path.join(dataset_variete_dir, variety)
        create_directory(target_variety_dir)

        for stage in os.listdir(variety_path):

            stage_path = os.path.join(variety_path, stage)

            if not os.path.isdir(stage_path):
                continue

            copy_images(stage_path, target_variety_dir, prefix=stage)

    print("\n✅ Dataset variété construit")

# --------------------------
# Gestion annotations
# --------------------------
def extract_annotations(zip_path, extract_dir):
    create_directory(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def copy_annotations(source_dir, target_dir, raw_dir):
    create_directory(target_dir)
    for file in os.listdir(source_dir):
        src_file = os.path.join(source_dir, file)
        if os.path.isfile(src_file):
            shutil.copy(src_file, os.path.join(target_dir, file))
    shutil.rmtree(raw_dir)

# ============================================================
def merge_images(source_dir, target_dir, image_ext = list(EXTENSIONS)):
    """Fusionner toutes les images dans un seul dossier"""

    os.makedirs(target_dir, exist_ok=True)

    img_count = 0

    print("\n📦 Fusion des images...")

    for root, dirs, files in os.walk(source_dir):

        for file in files:

            if Path(file).suffix.lower() in image_ext:

                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)

                if not os.path.exists(dst_path):

                    shutil.copy(src_path, dst_path)
                    img_count += 1

                    if img_count % 100 == 0:
                        print(f"    🔔 {img_count} images copiées")

                else:
                    print(f"⚠️ Image déjà existante ignorée : {file}")

    print(f"\n✅ {img_count} images copiées dans {target_dir}")
