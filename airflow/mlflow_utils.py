# airflow/mlflow_utils.py
import os
import mlflow
import logging
import socket
import time
from contextlib import contextmanager
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------
def wait_for_mlflow(tracking_uri, timeout=10):
    """
    Attend que MLflow soit disponible
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{tracking_uri}/health", timeout=2, verify=False)
            if response.status_code == 200:
                logger.info(f"✅ MLflow est prêt: {tracking_uri}")
                return True
        except:
            pass
        time.sleep(1)
    return False

# --------------------------------------------------------------------------------------------------------
def get_mlflow_tracking_uri():
    """
    Obtient l'URI de tracking MLflow avec résolution d'hôte
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    
    hosts_to_try = [tracking_uri]
    
    # Essayer de résoudre l'IP
    try:
        if "://" in tracking_uri:
            scheme, rest = tracking_uri.split("://")
            host_port = rest.split("/")[0]
            host = host_port.split(":")[0]
            ip = socket.gethostbyname(host)
            ip_uri = f"{scheme}://{ip}:5000"
            hosts_to_try.append(ip_uri)
            logger.info(f"IP résolue pour {host}: {ip}")
    except Exception as e:
        logger.warning(f"Impossible de résoudre l'hôte: {e}")
    
    # Essayer avec localhost
    if "://" in tracking_uri:
        scheme = tracking_uri.split("://")[0]
        local_uri = f"{scheme}://localhost:5000"
        hosts_to_try.append(local_uri)
    
    # Tester chaque URI
    for uri in hosts_to_try:
        logger.info(f"Test de connexion à MLflow: {uri}")
        if wait_for_mlflow(uri, timeout=2):
            logger.info(f"✅ URI MLflow valide trouvée: {uri}")
            return uri
    
    logger.warning(f"Aucune URI MLflow fonctionnelle, utilisation de: {tracking_uri}")
    return tracking_uri

# --------------------------------------------------------------------------------------------------------
def initialize_mlflow(tracking_uri=None, clean_on_error=True):
    """
    Initialise MLflow avec gestion des erreurs
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            tracking_uri = get_mlflow_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
        
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        os.environ["MLFLOW_TRACKING_IGNORE_TLS"] = "true"
        os.environ["MLFLOW_DISABLE_HOSTNAME_VALIDATION"] = "true"
        
        response = requests.get(f"{tracking_uri}/health", timeout=5, verify=False)
        if response.status_code == 200:
            logger.info(f"✅ MLflow initialisé avec succès: {tracking_uri}")
            return True
        else:
            logger.warning(f"⚠️ MLflow répond avec code {response.status_code}")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur d'initialisation MLflow: {e}")
        return False

# --------------------------------------------------------------------------------------------------------
@contextmanager
def safe_mlflow_run(experiment_name, run_name):
    """
    Context manager sécurisé pour les runs MLflow.
    Évite 'generator didn't stop after throw()' en ne yieldant pas après exception.
    """
    run = None
    try:
        tracking_uri = mlflow.get_tracking_uri()
        if not initialize_mlflow(tracking_uri):
            logger.warning("⚠️ MLflow non disponible, exécution sans logging")
            yield None
            return

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                exp_id = mlflow.create_experiment(experiment_name)
                logger.info(f"✅ Expérience créée: {experiment_name} (ID: {exp_id})")
            else:
                exp_id = experiment.experiment_id
                logger.info(f"✅ Expérience existante: {experiment_name}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur Experience: {e}")
            exp_id = None

        run = mlflow.start_run(run_name=run_name, experiment_id=exp_id)
        logger.info(f"✅ Run MLflow démarré: {run.info.run_id}")
        
        yield run
        
        mlflow.end_run(status='FINISHED')

    except Exception as e:
        logger.error(f"❌ Erreur critique MLflow: {e}")
        if run:
            try:
                mlflow.end_run(status='FAILED')
            except:
                pass
        logger.warning("⚠️ Erreur interceptée par safe_mlflow_run")
        return
