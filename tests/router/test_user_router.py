"""
Tests unitaires pour les routes d'historique (version simplifiée)
"""

import pytest
from fastapi import HTTPException
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.routers.user_router import user_router


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock()


@pytest.fixture
def mock_current_user():
    """Mock de l'utilisateur connecté"""
    user = Mock()
    user.id = 1
    user.role = "USER"
    return user


# =========================================================
# TESTS POUR GET_USER_HISTORY
# =========================================================

def test_get_user_history_success(mock_db):
    """
    Test 1: Récupération réussie de l'historique
    """
    # Arrange
    user_id = 1
    mock_histories = [Mock(), Mock()]
    
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.count.return_value = 2
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_histories
    
    mock_db.query.return_value = mock_query
    
    # Act
    result = user_router.routes[0].endpoint(
        user_id=user_id,
        page=1,
        limit=10,
        variety=None,
        maturity=None,
        date_from=None,
        date_to=None,
        db=mock_db
    )
    
    # Assert
    assert result["total"] == 2
    assert len(result["data"]) == 2
    assert result["page"] == 1
    assert result["limit"] == 10


def test_get_user_history_with_filters(mock_db):
    """
    Test 2: Récupération avec filtres
    """
    # Arrange
    user_id = 1
    variety = "Majhoul"
    date_from = datetime(2024, 1, 1)
    date_to = datetime(2024, 1, 31)
    
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.count.return_value = 1
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
    
    mock_db.query.return_value = mock_query
    
    # Act
    result = user_router.routes[0].endpoint(
        user_id=user_id,
        page=1,
        limit=10,
        variety=variety,
        maturity=None,
        date_from=date_from,
        date_to=date_to,
        db=mock_db
    )
    
    # Assert
    assert result["total"] == 1
    # Vérifier que le filtre a été appliqué
    mock_query.filter.assert_called()


def test_get_user_history_pagination(mock_db):
    """
    Test 3: Vérification de la pagination
    """
    # Arrange
    user_id = 1
    page = 2
    limit = 5
    
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.count.return_value = 12
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
    
    mock_db.query.return_value = mock_query
    
    # Act
    result = user_router.routes[0].endpoint(
        user_id=user_id,
        page=page,
        limit=limit,
        variety=None,
        maturity=None,
        date_from=None,
        date_to=None,
        db=mock_db
    )
    
    # Assert
    assert result["page"] == page
    assert result["limit"] == limit
    assert result["total"] == 12


# =========================================================
# TESTS POUR GET_MY_STATS
# =========================================================

@pytest.mark.anyio
async def test_get_my_stats_success(mock_db, mock_current_user):
    """
    Test 1: Récupération réussie des stats utilisateur
    """
    # Arrange
    expected_stats = {"user": {"id": 1}, "summary": {"total": 10}}
    
    with patch('app.routers.user_router.get_user_statistics') as mock_get:
        mock_get.return_value = expected_stats
        
        # Act
        result = await user_router.routes[1].endpoint(
            current_user=mock_current_user,
            db=mock_db
        )
        
        # Assert
        assert result == expected_stats
        mock_get.assert_called_once_with(
            db=mock_db,
            user_id=mock_current_user.id,
            requesting_user_id=mock_current_user.id,
            requesting_user_role=mock_current_user.role
        )


@pytest.mark.anyio
async def test_get_my_stats_user_not_found(mock_db, mock_current_user):
    """
    Test 2: Utilisateur non trouvé
    """
    # Arrange
    with patch('app.routers.user_router.get_user_statistics') as mock_get:
        mock_get.side_effect = ValueError("User not found")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc:
            await user_router.routes[1].endpoint(
                current_user=mock_current_user,
                db=mock_db
            )
        
        assert exc.value.status_code == 404


@pytest.mark.anyio
async def test_get_my_stats_error(mock_db, mock_current_user):
    """
    Test 3: Gestion d'erreur générale
    """
    # Arrange
    with patch('app.routers.user_router.get_user_statistics') as mock_get:
        mock_get.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc:
            await user_router.routes[1].endpoint(
                current_user=mock_current_user,
                db=mock_db
            )
        
        assert exc.value.status_code == 500