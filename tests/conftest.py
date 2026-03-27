# tests/conftest.py
import pytest 
from unittest.mock import Mock, patch
from app.core.config import settings
import torch 

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Configuration pour les tests d'images
@pytest.fixture(autouse=True)
def mock_torch_imports():
    """Mock torch pour éviter les problèmes de GPU"""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.device', return_value=torch.device('cpu')):
            yield