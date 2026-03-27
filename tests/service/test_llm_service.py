"""
Tests unitaires pour le service de génération de rapport
"""

import pytest
from unittest.mock import Mock, patch
import requests

from app.services.llm_service import generate_report
from app.core.config import settings


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def sample_results():
    """Exemple de résultats de détection"""
    return {
        "total_detections": 150,
        "varieties": {
            "Majhoul": 45,
            "Boufaguos": 60,
            "Kholt": 25,
            "Bouisthami": 20
        },
        "maturity": {
            "Stage1": 30,
            "Stage2": 50,
            "Stage3": 40,
            "Stage4": 30
        }
    }


@pytest.fixture
def mock_llm_response():
    """Mock de la réponse du LLM"""
    return {
        "response": """
## 📊 Synthèse
- Total : 150 dattes

## 🌱 Répartition par variété
- Boufaguos : 60 (40%)
- Majhoul : 45 (30%)
- Kholt : 25 (17%)
- Bouisthami : 20 (13%)

## 🍂 Répartition par maturité
- Stage 2 (Khalal) : 50 (33%)
- Stage 3 (Rutab) : 40 (27%)
- Stage 1 (Immature) : 30 (20%)
- Stage 4 (Tamar) : 30 (20%)

## 💡 Observations
- La variété Boufaguos est majoritaire avec 40% des dattes
- Le stade Khalal prédomine avec 33% des dattes
- Distribution équilibrée entre les différents stades de maturité
"""
    }


# =========================================================
# TESTS POUR GENERATE_REPORT
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_generate_report_success(sample_results, mock_llm_response):
    """
    Test 1: Génération réussie du rapport avec réponse LLM valide
    """
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_llm_response
    
    with patch('requests.post') as mock_post:
        mock_post.return_value = mock_response
        
        # Act
        result = generate_report(sample_results)
        
        # Assert
        assert result == mock_llm_response["response"]
        mock_post.assert_called_once()
        
        # Vérifier les paramètres de l'appel
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == settings.OLLAMA_MODEL
        assert call_args["json"]["stream"] is False
        assert "prompt" in call_args["json"]


# --------------------------------------------------------------------------------------------------------
def test_generate_report_http_error(sample_results):
    """
    Test 2: Erreur HTTP lors de l'appel au LLM
    """
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    with patch('requests.post') as mock_post:
        mock_post.return_value = mock_response
        
        # Act
        report = generate_report(sample_results)
        
        # Assert
        assert "Erreur Modèle" in report


# --------------------------------------------------------------------------------------------------------
def test_generate_report_missing_response_key(sample_results):
    """
    Test 3: Réponse LLM sans la clé 'response'
    """
    # Arrange
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "Something went wrong"}
    
    with patch('requests.post') as mock_post:
        mock_post.return_value = mock_response
        
        # Act
        report = generate_report(sample_results)
        
        # Assert
        assert "Erreur Serveur" in report