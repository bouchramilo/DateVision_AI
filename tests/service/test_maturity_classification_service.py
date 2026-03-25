"""
Tests unitaires pour le service de classification de maturité
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from app.services import maturity_service
from app.services.maturity_service import (
    get_maturity_model,
    predict_maturity,
    CLASSES,
    DEVICE
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def real_image():
    """Crée une vraie image PIL pour les tests"""
    return Image.new('RGB', (224, 224), color='red')


@pytest.fixture
def mock_model():
    """Mock du modèle de classification"""
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    
    # Simuler une prédiction
    mock_output = torch.tensor([[2.5, 1.2, 3.8, 0.5]])
    mock_model.return_value = mock_output
    
    return mock_model


# =========================================================
# TESTS POUR GET_MATURITY_MODEL
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_maturity_model_success():
    """
    Test 1: Chargement réussi du modèle de maturité
    """
    maturity_service._model = None
    
    mock_model_path = "/fake/path/to/maturity_model.pth"
    
    # Mock des dépendances
    with patch('os.path.exists', return_value=True):
        with patch('torch.load') as mock_torch_load:
            with patch('torchvision.models.googlenet') as mock_googlenet:
                with patch('torch.nn.Sequential') as mock_sequential:
                    
                    mock_state_dict = {"layer1.weight": torch.randn(10, 10)}
                    mock_torch_load.return_value = {"model_state_dict": mock_state_dict}
                    
                    mock_googlenet_instance = Mock()
                    mock_googlenet.return_value = mock_googlenet_instance
                    
                    result = get_maturity_model(mock_model_path)
                    
                    assert result is not None
                    mock_torch_load.assert_called_once_with(mock_model_path, map_location=DEVICE)
                    mock_googlenet.assert_called_once_with(weights=None, aux_logits=True)
                    mock_googlenet_instance.load_state_dict.assert_called_once_with(mock_state_dict, strict=False)
                    mock_googlenet_instance.to.assert_called_once_with(DEVICE)
                    mock_googlenet_instance.eval.assert_called_once()


# --------------------------------------------------------------------------------------------------------
def test_get_maturity_model_singleton():
    """
    Test 2: Le modèle est chargé une seule fois (Singleton)
    """
    maturity_service._model = None
    
    mock_model_path = "/fake/path/to/maturity_model.pth"
    
    with patch('os.path.exists', return_value=True):
        with patch('torch.load') as mock_torch_load:
            with patch('torchvision.models.googlenet') as mock_googlenet:
                mock_torch_load.return_value = {"model_state_dict": {}}
                mock_googlenet.return_value = Mock()
                
                model1 = get_maturity_model(mock_model_path)
                
                model2 = get_maturity_model(mock_model_path)
                
                assert mock_torch_load.call_count == 1
                assert model1 is model2
                assert model1 == mock_googlenet.return_value


# =========================================================
# TESTS POUR PREDICT_MATURITY
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_predict_maturity_success(real_image):
    """
    Test 3: Prédiction réussie de la maturité
    """
    mock_model = Mock()
    
    mock_output = torch.tensor([[2.5, 1.2, 3.8, 0.5]])
    mock_model.return_value = mock_output
    
    mock_tensor = torch.randn(1, 3, 224, 224)
    
    with patch('app.services.maturity_service.get_maturity_model', return_value=mock_model):
        with patch('app.services.maturity_service.preprocess_for_classification', return_value=mock_tensor):
            with patch('torch.no_grad'):
                
                result = predict_maturity(real_image)
                
                assert isinstance(result, dict)
                assert "class_id" in result
                assert "class_name" in result
                assert "confidence" in result
                assert result["class_id"] == 2
                assert result["class_name"] == "S3"
                assert isinstance(result["confidence"], float)
                assert 0 <= result["confidence"] <= 1


# --------------------------------------------------------------------------------------------------------
def test_predict_maturity_with_aux_logits(real_image):
    """
    Test 4: Prédiction avec aux_logits (modèle avec sorties multiples)
    """
    mock_model = Mock()
    
    main_output = torch.tensor([[1.5, 2.8, 0.9, 3.2]])
    aux_output = torch.tensor([[1.2, 2.5, 1.1, 2.9]])
    mock_model.return_value = (main_output, aux_output)
    
    mock_tensor = torch.randn(1, 3, 224, 224)
    
    with patch('app.services.maturity_service.get_maturity_model', return_value=mock_model):
        with patch('app.services.maturity_service.preprocess_for_classification', return_value=mock_tensor):
            with patch('torch.no_grad'):
                
                result = predict_maturity(real_image)
                
                assert result["class_id"] == 3
                assert result["class_name"] == "S4"
                assert isinstance(result["confidence"], float)


# --------------------------------------------------------------------------------------------------------
def test_predict_maturity_confidence_values(real_image):
    """
    Test 5: Vérification des valeurs de confiance (softmax)
    """
    
    mock_model = Mock()
    
    logits = torch.tensor([[10.0, 5.0, 0.0, -5.0]])
    mock_model.return_value = logits
    
    mock_tensor = torch.randn(1, 3, 224, 224)
    
    with patch('app.services.maturity_service.get_maturity_model', return_value=mock_model):
        with patch('app.services.maturity_service.preprocess_for_classification', return_value=mock_tensor):
            with patch('torch.no_grad'):
                
                result = predict_maturity(real_image)
                
                assert result["class_id"] == 0
                assert result["class_name"] == "S1"
                assert result["confidence"] > 0.99