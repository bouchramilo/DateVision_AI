# setup_airflow.ps1
Write-Host "Creating Airflow directory structure..." -ForegroundColor Green

# Créer les dossiers nécessaires
$folders = @(
    "airflow\dags",
    "airflow\dags\functions",
    "airflow\dags\tasks",
    "airflow\dags\config",
    "airflow\logs",
    "airflow\plugins",
    "airflow\tasks",
    "airflow\functions",
    "airflow\config",
    "data\dataset_classification\dataset_variete",
    "data\dataset_classification\dataset_maturite",
    "analyse_configuration\variete_analyse",
    "analyse_configuration\maturity_analyse",
    "models"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Write-Host "  Created: $folder" -ForegroundColor Yellow
}

# Créer les fichiers __init__.py
$initFiles = @(
    "airflow\dags\__init__.py",
    "airflow\dags\functions\__init__.py",
    "airflow\dags\tasks\__init__.py",
    "airflow\dags\config\__init__.py",
    "airflow\tasks\__init__.py",
    "airflow\functions\__init__.py",
    "airflow\config\__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path $file | Out-Null
    Write-Host "  Created: $file" -ForegroundColor Yellow
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Now you can run: docker compose -f docker/docker-compose.yml up -d --build" -ForegroundColor Cyan