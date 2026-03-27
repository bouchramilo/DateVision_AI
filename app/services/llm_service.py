import requests
import time
from typing import Any

from app.core.config import settings
from app.core.metrics import (
    LLM_REQUESTS_TOTAL,
    LLM_LATENCY,
    LLM_ERRORS,
    LLM_RESPONSE_LENGTH,
    LLM_QUALITY_SCORE,
)

# =========================================================
# 🔹 CORE REPORT GENERATION
# =========================================================

def generate_report(results: Any) -> str:
    """
    Envoie les résultats de détection à Ollama (LLM) pour générer 
    un rapport structuré en français.
    """
    prompt = build_prompt(results)
    LLM_REQUESTS_TOTAL.inc()
    start = time.perf_counter()

    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
    except Exception as exc:
        LLM_ERRORS.inc()
        return "## ⚠️ Service Indisponible\nLe modèle linguistique est temporairement indisponible."

    # Calcul de latence et validation HTTP
    llm_duration = time.perf_counter() - start
    LLM_LATENCY.observe(llm_duration)

    if response.status_code != 200:
        LLM_ERRORS.inc()
        return "## ⚠️ Erreur Modèle\nLe modèle linguistique a rencontré une erreur. Les mesures restent valides."

    resp_json = response.json()

    if "response" not in resp_json:
        LLM_ERRORS.inc()
        return "## ⚠️ Erreur Serveur\nLe service a retourné une réponse invalide."

    report_text = resp_json["response"]

    LLM_RESPONSE_LENGTH.observe(len(report_text))
    quality = min(len(report_text) / 1000.0, 1.0)
    LLM_QUALITY_SCORE.set(quality)

    return report_text


# =========================================================
# 🔹 PROMPT BUILDER
# =========================================================

def build_prompt(results: Any) -> str:
    """
    Construit le prompt détaillé pour l'analyse des dattes.
    """
    return f"""
    Analyse les données de détection suivantes :

    {results}

    Génère un rapport structuré avec les sections suivantes :
    1. **Total** : Nombre total de dattes détectées
    2. **Variétés** : Distribution par variété (pourcentage et nombre)
    3. **Maturité** : Répartition par stade de maturité (pourcentage et nombre)
    4. **Observations** : Synthèse des tendances principales
    
    • Variétés de dattes : Majhoul, Boufaguos, Kholt, et Bouisthami.
    • Étapes de maturité : Stage 1 (Immature), Stage 2 (Khalal), Stage 3 (Rutab), Stage 4 (Tamar).

    Directives :
    - Style : concis, technique et factuel
    - Arrondir les scores de confiance à 1 décimale
    - Mettre en évidence les points notables (variété dominante, stade majoritaire)
    - Pas d'introduction ni de conclusion superflue
    - Uniquement en français

    Format attendu :

    ## 📊 Synthèse
    - Total : X dattes

    ## 🌱 Répartition par variété
    - Variété A : X (X%)
    - Variété B : X (X%)
    ...

    ## 🍂 Répartition par maturité
    - Stage 1 (Immature) : X (X%)
    - Stage 2 (Khalal) : X (X%)
    - Stage 3 (Rutab) : X (X%)
    - Stage 4 (Tamar) : X (X%)
    
    ## 💡 Observations
    - [Point clé 1]
    - [Point clé 2]
    """