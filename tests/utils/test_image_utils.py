"""
Tests unitaires pour les utilitaires d'image
"""

import pytest
import base64
import cv2
import numpy as np
from unittest.mock import patch, Mock

from app.utils.image_util import image_to_base64


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def sample_image():
    """Crée une image numpy de test"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Crée une image en niveaux de gris"""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


# =========================================================
# TESTS POUR IMAGE_TO_BASE64
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_image_to_base64_success(sample_image):
    """
    Test 1: Conversion réussie d'une image numpy en base64
    """
    # Act
    result = image_to_base64(sample_image)
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Vérifier que c'est bien du base64 valide
    try:
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    except Exception as e:
        pytest.fail(f"Le résultat n'est pas un base64 valide: {e}")


# --------------------------------------------------------------------------------------------------------
def test_image_to_base64_with_grayscale(sample_grayscale_image):
    """
    Test 2: Conversion d'une image en niveaux de gris
    """
    # Act
    result = image_to_base64(sample_grayscale_image)
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Vérifier que c'est du base64 valide
    try:
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    except Exception:
        pytest.fail("Le résultat n'est pas un base64 valide")


# --------------------------------------------------------------------------------------------------------
def test_image_to_base64_with_mock(sample_image):
    """
    Test 3: Vérification des appels avec mock (imencode)
    """
    # Arrange
    mock_buffer = np.array([1, 2, 3])
    
    with patch('cv2.imencode') as mock_imencode:
        mock_imencode.return_value = (True, mock_buffer)
        
        # Act
        result = image_to_base64(sample_image)
        
        # Assert
        mock_imencode.assert_called_once_with(".jpg", sample_image)
        
        # Vérifier que le résultat est le base64 du buffer mocké
        expected = base64.b64encode(mock_buffer).decode("utf-8")
        assert result == expected
        
        
# --------------------------------------------------------------------------------------------------------