"""
Tests unitaires pour les utilitaires JWT avec mocks
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from app.security.manage_token import create_access_token
from app.core.config import settings


# =========================================================
# TESTS POUR CREATE_ACCESS_TOKEN AVEC MOCKS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_create_access_token_calls_jwt_encode():
    """
    Test 1: Vérifie que jwt.encode est appelé avec les bons paramètres
    """
    # Arrange
    subject = "user@example.com"
    mock_token = "mock.jwt.token"
    
    with patch('app.security.manage_token.jwt.encode') as mock_encode:
        mock_encode.return_value = mock_token
        
        # Act
        result = create_access_token(subject)
        
        # Assert
        assert result == mock_token
        mock_encode.assert_called_once()
        
        # Vérifier les arguments passés à jwt.encode
        args, kwargs = mock_encode.call_args
        assert args[1] == settings.SECRET_KEY
        assert kwargs["algorithm"] == settings.ALGORITHM
        
        # Vérifier le payload
        payload = args[0]
        assert payload["sub"] == subject
        assert "exp" in payload


# --------------------------------------------------------------------------------------------------------
def test_create_access_token_with_mocked_time():
    """
    Test 2: Vérifie que l'expiration est correctement calculée
    """
    # Arrange
    subject = "user@example.com"
    fixed_now = datetime(2024, 1, 1, 12, 0, 0)
    expected_exp = fixed_now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    with patch('app.security.manage_token.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = fixed_now
        
        with patch('app.security.manage_token.jwt.encode') as mock_encode:
            mock_encode.return_value = "token"
            
            # Act
            create_access_token(subject)
            
            # Assert
            payload = mock_encode.call_args[0][0]
            assert payload["exp"] == expected_exp
            assert payload["sub"] == subject


# --------------------------------------------------------------------------------------------------------
def test_create_access_token_custom_delta_mocked():
    """
    Test 3: Vérifie que l'expiration personnalisée est utilisée
    """
    # Arrange
    subject = "user@example.com"
    custom_delta = timedelta(minutes=15)
    fixed_now = datetime(2024, 1, 1, 12, 0, 0)
    expected_exp = fixed_now + custom_delta
    
    with patch('app.security.manage_token.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = fixed_now
        
        with patch('app.security.manage_token.jwt.encode') as mock_encode:
            mock_encode.return_value = "token"
            
            # Act
            create_access_token(subject, custom_delta)
            
            # Assert
            payload = mock_encode.call_args[0][0]
            assert payload["exp"] == expected_exp
            assert payload["sub"] == subject