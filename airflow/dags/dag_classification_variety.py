# airflow/dags/dag_classification_variety.py
"""
DAG spécifique pour la classification des variétés
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_config import VARIETY_CONFIG, MATURITY_CONFIG
from tasks.classification_tasks import (
    load_and_explore_data,
    split_dataset_task,
    prepare_datasets,
    create_balanced_loaders,
    train_model_task,
    evaluate_model_task
)


# ============================================================================
# DAG : Training model : classification by variety
# ============================================================================
@dag(
    dag_id='variety_classification_only',
    default_args={
        'owner': 'data_scientist',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Pipeline de classification des variétés de dattes',
    schedule_interval='@yearly',
    catchup=False,
    tags=['classification', 'variety', 'production']
)
def variety_classification_only():
    
    config = VARIETY_CONFIG
    
    # Définition des tâches
    explore_task = load_and_explore_data(config)
    split_task = split_dataset_task(config, explore_task)
    dataset_task = prepare_datasets(config)
    loaders_task = create_balanced_loaders(config, dataset_task)
    train_task = train_model_task(config, dataset_task, loaders_task)
    eval_task = evaluate_model_task(config, dataset_task, train_task)
    
    # Ordre d'exécution
    explore_task >> split_task >> dataset_task >> loaders_task >> train_task >> eval_task


# Création du DAG
variety_dag = variety_classification_only()