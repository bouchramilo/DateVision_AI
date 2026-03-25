"""
Tests unitaires pour le repository admin
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.repositories.admin_repository import get_admin_stats_repo
from app.models.user_model import User
from app.models.history_model import History
from app.models.result_model import Result


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock(spec=Session)


# =========================================================
# TESTS POUR GET_ADMIN_STATS_REPO
# =========================================================
# --------------------------------------------------------------------------------------------------------
def test_get_admin_stats_repo_success(mock_db):
    """
    Test 1: Récupération réussie des statistiques admin
    """
    with patch('app.repositories.admin_repository.datetime') as mock_datetime:
        fixed_now = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.utcnow.return_value = fixed_now
        
        # 1. Mock total_users
        mock_total_users = MagicMock()
        mock_total_users.scalar.return_value = 150
        
        # 2. Mock active_users
        mock_active_users = MagicMock()
        mock_active_users.filter.return_value.scalar.return_value = 45
        
        # 3. Mock total_queries
        mock_total_queries = MagicMock()
        mock_total_queries.scalar.return_value = 1200
        
        # 4. Mock total_detections
        mock_total_detections = MagicMock()
        mock_total_detections.scalar.return_value = 3500
        
        # 5. Mock top_varieties
        mock_varieties = [
            Mock(variety="Majhoul", count=500),
            Mock(variety="Boufagous", count=400),
            Mock(variety="Kholt", count=300)
        ]
        mock_top_varieties = MagicMock()
        mock_top_varieties.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = mock_varieties
        
        # 6. Mock top_maturities
        mock_maturities = [
            Mock(maturity="Stage 3", count=800),
            Mock(maturity="Stage 2", count=600),
            Mock(maturity="Stage 4", count=400)
        ]
        mock_top_maturities = MagicMock()
        mock_top_maturities.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = mock_maturities
        
        # Configurer les appels successifs à query()
        mock_db.query.side_effect = [
            mock_total_users,
            mock_active_users,
            mock_total_queries,
            mock_total_detections,
            mock_top_varieties,
            mock_top_maturities
        ]
        
        # Act
        result = get_admin_stats_repo(mock_db)
        
        # Assert
        assert result["total_users"] == 150
        assert result["active_users_count"] == 45
        assert result["total_queries"] == 1200
        assert result["total_detections"] == 3500
        assert len(result["top_varieties"]) == 3
        assert result["top_varieties"][0]["name"] == "Majhoul"
        assert result["top_varieties"][0]["count"] == 500
        assert len(result["top_maturities"]) == 3
        assert result["top_maturities"][0]["name"] == "Stage 3"
        assert result["top_maturities"][0]["count"] == 800


# --------------------------------------------------------------------------------------------------------# --------------------------------------------------------------------------------------------------------
def test_get_admin_stats_repo_empty_data(mock_db):
    """
    Test 2: Récupération des statistiques quand il n'y a pas de données
    """
    # 1. total_users
    mock_total_users = MagicMock()
    mock_total_users.scalar.return_value = 0
    
    # 2. active_users
    mock_active_users = MagicMock()
    mock_active_users.filter.return_value.scalar.return_value = 0
    
    # 3. total_queries
    mock_total_queries = MagicMock()
    mock_total_queries.scalar.return_value = 0
    
    # 4. total_detections
    mock_total_detections = MagicMock()
    mock_total_detections.scalar.return_value = 0
    
    # 5. top_varieties
    mock_top_varieties = MagicMock()
    mock_top_varieties.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
    
    # 6. top_maturities
    mock_top_maturities = MagicMock()
    mock_top_maturities.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
    
    mock_db.query.side_effect = [
        mock_total_users,
        mock_active_users,
        mock_total_queries,
        mock_total_detections,
        mock_top_varieties,
        mock_top_maturities
    ]
    
    # Act
    result = get_admin_stats_repo(mock_db)
    
    # Assert
    assert result["total_users"] == 0
    assert result["active_users_count"] == 0
    assert result["total_queries"] == 0
    assert result["total_detections"] == 0
    assert result["top_varieties"] == []
    assert result["top_maturities"] == []


# --------------------------------------------------------------------------------------------------------# --------------------------------------------------------------------------------------------------------
def test_get_admin_stats_repo_structure(mock_db):
    """
    Test 3: Vérification de la structure des données retournées
    """
    # Mock des données pour la structure
    mock_scalar = MagicMock()
    mock_scalar.scalar.return_value = 10
    
    mock_active = MagicMock()
    mock_active.filter.return_value.scalar.return_value = 5
    
    mock_list = MagicMock()
    mock_list.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
    
    mock_db.query.side_effect = [
        mock_scalar, # total_users
        mock_active, # active_users
        mock_scalar, # total_queries
        mock_scalar, # total_detections
        mock_list,   # top_varieties
        mock_list    # top_maturities
    ]
    
    # Act
    result = get_admin_stats_repo(mock_db)
    
    # Assert - Vérifier la structure
    assert isinstance(result, dict)
    assert "total_users" in result
    assert "active_users_count" in result
    assert "total_queries" in result
    assert "total_detections" in result
    assert "top_varieties" in result
    assert "top_maturities" in result
    
    # Vérifier le format des top_varieties
    assert isinstance(result["top_varieties"], list)
    for item in result["top_varieties"]:
        assert "name" in item
        assert "count" in item
    
    # Vérifier le format des top_maturities
    assert isinstance(result["top_maturities"], list)
    for item in result["top_maturities"]:
        assert "name" in item
        assert "count" in item