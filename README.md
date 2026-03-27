# 🌴 DateVision AI – Détection & Classification des Dattes avec YOLOv8

## 📌 Description du projet

**DateVision AI** est une solution complète d’intelligence artificielle permettant la **détection et la classification automatique des dattes à partir d’images** en utilisant des techniques avancées de **Deep Learning**, notamment **YOLOv8**.

Ce projet a été réalisé dans le cadre du **Projet de Fin d’Études (YouCode)** et couvre l’ensemble du cycle de vie d’un produit IA :  
👉 collecte des données → entraînement → API → interface → déploiement → monitoring.

L’objectif est de répondre à des besoins réels dans :
- 🌱 Smart Agriculture  
- 🏭 Contrôle qualité agroalimentaire  
- 🤖 Automatisation du tri des dattes  

Le système est enrichi par un **LLM (Large Language Model)** pour fournir :
- des **explications intelligentes**
- des **recommandations métier**
- des **rapports automatiques**

---

## 🚀 Fonctionnalités principales

### 🔍 Computer Vision
- Détection des dattes dans une image (bounding boxes)
- Classification multi-classes :
  - Variété
  - Stade de maturité
- Affichage des résultats annotés avec scores

### 🧠 Intelligence augmentée (LLM)
- Interprétation des résultats YOLO
- Génération d’explications en langage naturel
- Recommandations (tri, stockage, qualité)

### 🌐 API & Backend
- API REST avec **FastAPI**
- Authentification sécurisée (**JWT**)
- Historique des prédictions
- Validation des données avec **Pydantic**

### 📊 Monitoring & MLOps
- Tracking des expériences avec **MLflow**
- Monitoring via **Prometheus & Grafana**
- Pipelines automatisés avec **Apache Airflow**
- CI/CD avec **GitHub Actions**

### 🖥️ Fonctionnalités avancées
- Upload d’images
- Dashboard de visualisation
- Logs & métriques du modèle
- Tests unitaires & intégration

---

## 🗂️ Structure du projet

```bash
DateVision_AI/
├── .github/workflows/        # CI/CD (GitHub Actions)
├── Conception/               # Diagrammes UML (Use Case, Classes)
│
├── airflow/                  # Pipelines de données & ML (Airflow)
│   ├── dags/                 # DAGs (ETL, détection, classification)
│   ├── tasks/                # Tâches Airflow
│   ├── functions/            # Pipelines ML & utils
│
├── app/                      # Backend FastAPI
│   ├── routers/              # Endpoints API
│   ├── services/             # Logique métier (YOLO, LLM, etc.)
│   ├── models/               # Modèles base de données
│   ├── schemas/              # Validation Pydantic
│   ├── repositories/         # Accès aux données
│   ├── security/             # JWT & hashing
│   └── core/                 # Config, logs, exceptions
│
├── models/                   # Modèles entraînés (YOLO, classification)
├── notebooks/                # Expérimentation & entraînement
├── analyse_configuration/    # Résultats & métriques d’évaluation
│
├── monitoring/               # Prometheus & Grafana
├── docker/                   # Docker & docker-compose
├── tests/                    # Tests unitaires & intégration
│
├── requirements.txt          # Dépendances Python
└── README.md
```

## 🛠️ Technologies utilisées
 🔹 IA & Data

* Python
* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* NumPy / Pandas

🔹 Backend & API

* FastAPI
* Pydantic
* JWT Authentication

🔹 MLOps & Data Engineering

* MLflow
* Apache Airflow
* GitHub Actions
* Docker & Docker Compose
  
🔹 Monitoring

* Prometheus
* Grafana

🔹 LLM & IA Générative

* Ollama
* Prompt Engineering

## ⚙️ Installation et exécution

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/bouchramilo/DateVision_AI.git
cd DateVision_AI
```

### 2️⃣ Configurer les variables d’environnement

Créer un fichier .env basé sur .env.example :
```bash
cp .env.example .env
```

Configurer :

* Base de données
* Clés API LLM
* Paramètres MLflow


### 3️⃣ Lancer le projet avec Docker

```bash
docker compose -f docker\docker-compose.yml up -d --build
```

### 5️⃣ Arrêter les services

```bash
docker compose -f docker\docker-compose.yml down
```

---

## 🚀 Les endpoints principales : 

(je vais ajouter les endpoint et leur documentation ici)


--- 

## 🚀 Frontend de projet : 

Lien de repo de frontend : 
```bash
https://github.com/bouchramilo/DateVision_AI_frontend
```

--- 

Merci 🌴🌴😊
