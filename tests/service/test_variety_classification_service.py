"""
Tests unitaires pour le service de classification de variété
"""

import pytest
from unittest.mock import Mock, patch
import torch
from PIL import Image

from app.services.variety_service import predict_variety


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def real_image():
    """Crée une vraie image PIL pour les tests"""
    return Image.new('RGB', (224, 224), color='red')


# =========================================================
# TESTS POUR PREDICT_VARIETY
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_predict_variety_success(real_image):
    """
    Test 1: Prédiction réussie de la variété
    """

    mock_model = Mock()
    
    mock_output = torch.tensor([[3.5, 1.2, 0.8, 0.5]])
    mock_model.return_value = mock_output
    
    mock_tensor = torch.randn(1, 3, 224, 224)
    
    with patch('app.services.variety_service.get_variety_model', return_value=mock_model):
        with patch('app.services.variety_service.preprocess_for_classification', return_value=mock_tensor):
            with patch('torch.no_grad'):
                
                # Act
                result = predict_variety(real_image)
                
                # Assert
                assert result["class_id"] == 0
                assert result["class_name"] == "Boufagous"
                assert isinstance(result["confidence"], float)
                assert 0 <= result["confidence"] <= 1


# --------------------------------------------------------------------------------------------------------
def test_predict_variety_with_aux_logits(real_image):
    """
    Test 2: Prédiction avec modèle ayant aux_logits (sortie tuple)
    """

    mock_model = Mock()
    
    main_output = torch.tensor([[1.5, 3.8, 0.9, 1.2]])
    aux_output = torch.tensor([[1.2, 2.5, 1.1, 0.9]])
    mock_model.return_value = (main_output, aux_output)
    
    mock_tensor = torch.randn(1, 3, 224, 224)
    
    with patch('app.services.variety_service.get_variety_model', return_value=mock_model):
        with patch('app.services.variety_service.preprocess_for_classification', return_value=mock_tensor):
            with patch('torch.no_grad'):
                
                # Act
                result = predict_variety(real_image)
                
                # Assert
                assert result["class_id"] == 1  # Index 1 = 'Boumajhoul'
                assert result["class_name"] == "Boumajhoul"
                assert isinstance(result["confidence"], float)


# --------------------------------------------------------------------------------------------------------
def test_predict_variety_high_confidence(real_image):
    """
    Test 3: Vérification de la confiance pour une prédiction forte
    """

    mock_model = Mock()
    
    logits = torch.tensor([[0.5, 0.3, 12.0, 0.1]])
    mock_model.return_value = logits
    
    mock_tensor = torch.randn(1, 3, 224, 224)
    
    with patch('app.services.variety_service.get_variety_model', return_value=mock_model):
        with patch('app.services.variety_service.preprocess_for_classification', return_value=mock_tensor):
            with patch('torch.no_grad'):
                
                # Act
                result = predict_variety(real_image)
                
                # Assert
                assert result["class_id"] == 2  # Index 2 = 'bouisthami'
                assert result["class_name"] == "bouisthami"
                # La confiance devrait être très proche de 1 (> 99.9%)
                assert result["confidence"] > 0.999