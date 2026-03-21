import requests
from app.core.config import settings

# ============================================
# Geerate rapport 
# ============================================
def generate_report(results):
    prompt = build_prompt(results)

    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        raise ValueError(f"Erreur LLM : {response.text}")
    resp_json = response.json()
    if "response" not in resp_json:
        raise ValueError(f"Clé 'response' manquante : {resp_json}")
    return resp_json["response"]


# ============================================
# Prompt function 
# ============================================
def build_prompt(results):
    """
    Construire prompt propre pour LLM
    """

    prompt = f"""
    Analyse les données de détection suivantes :

    {results}

    Génère un rapport structuré avec les sections suivantes :
    1. **Total** : Nombre total de dattes détectées
    2. **Variétés** : Distribution par variété (pourcentage et nombre)
    3. **Maturité** : Répartition par stade de maturité (pourcentage et nombre)
    4. **Observations** : Synthèse des tendances principales

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
    - Stage 1 (Verte) : X (X%)
    - Stage 2 (Mûre) : X (X%)
    ...

    ## 💡 Observations
    - [Point clé 1]
    - [Point clé 2]
    """

    return prompt