"""
Tests unitaires pour le service d'historique
"""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.history_service import get_user_statistics
from app.models.user_model import User


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock(spec=Session)


@pytest.fixture
def mock_user():
    """Mock d'un utilisateur"""
    user = Mock(spec=User)
    user.id = 1
    user.username = "testuser"
    user.role = "USER"
    return user


@pytest.fixture
def sample_stats():
    """Exemple de statistiques"""
    return {
        "total_predictions": 10,
        "successful_predictions": 8,
        "failed_predictions": 2,
        "average_confidence": 0.85,
        "most_common_disease": "flu",
        "predictions_by_day": []
    }


# =========================================================
# TESTS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_statistics_success_own_stats(mock_db, mock_user, sample_stats):
    """
    Test: Un utilisateur peut voir ses propres statistiques
    """
    user_id = 1
    requesting_user_id = 1
    requesting_user_role = "USER"
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    with patch('app.services.history_service.get_user_stats') as mock_get_stats:
        mock_get_stats.return_value = sample_stats
        
        result = get_user_statistics(
            db=mock_db,
            user_id=user_id,
            requesting_user_id=requesting_user_id,
            requesting_user_role=requesting_user_role
        )
        
        assert result == sample_stats
        mock_get_stats.assert_called_once_with(mock_db, user_id)


# --------------------------------------------------------------------------------------------------------
def test_get_user_statistics_admin_access(mock_db, mock_user, sample_stats):
    """
    Test: Un admin peut voir les statistiques d'un autre utilisateur
    """
    user_id = 1
    requesting_user_id = 2
    requesting_user_role = "ADMIN"
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    with patch('app.services.history_service.get_user_stats') as mock_get_stats:
        mock_get_stats.return_value = sample_stats
        
        result = get_user_statistics(
            db=mock_db,
            user_id=user_id,
            requesting_user_id=requesting_user_id,
            requesting_user_role=requesting_user_role
        )
        
        assert result == sample_stats
        mock_get_stats.assert_called_once_with(mock_db, user_id)


# --------------------------------------------------------------------------------------------------------
def test_get_user_statistics_unauthorized(mock_db, mock_user):
    """
    Test: Un utilisateur normal ne peut pas voir les stats d'un autre utilisateur
    """
    user_id = 2
    requesting_user_id = 1
    requesting_user_role = "USER"
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    with pytest.raises(PermissionError) as exc_info:
        get_user_statistics(
            db=mock_db,
            user_id=user_id,
            requesting_user_id=requesting_user_id,
            requesting_user_role=requesting_user_role
        )
    
    assert "Vous n'avez pas les droits" in str(exc_info.value)
    
    
# --------------------------------------------------------------------------------------------------------
