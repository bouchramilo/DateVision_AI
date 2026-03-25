"""
Tests unitaires pour les routes admin
"""

import pytest
from fastapi import HTTPException, status
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.routers.admin_router import admin_router
from app.models.user_model import User as UserModel
from app.schemas.user_schema import User as UserSchema


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock()


@pytest.fixture
def mock_current_admin():
    """Mock d'un administrateur connecté"""
    admin = Mock(spec=UserModel)
    admin.id = 1
    admin.email = "admin@example.com"
    admin.role = "ADMIN"
    admin.is_active = True
    return admin


@pytest.fixture
def mock_current_user():
    """Mock d'un utilisateur normal"""
    user = Mock(spec=UserModel)
    user.id = 2
    user.email = "user@example.com"
    user.role = "USER"
    user.is_active = True
    return user


@pytest.fixture
def sample_users():
    """Liste d'utilisateurs mockés"""
    return [
        UserSchema(
            id=1,
            email="admin@example.com",
            username="admin",
            role="ADMIN",
            is_active=True,
            created_at=datetime(2024, 1, 1)
        ),
        UserSchema(
            id=2,
            email="user@example.com",
            username="user",
            role="USER",
            is_active=True,
            created_at=datetime(2024, 1, 2)
        )
    ]


# =========================================================
# TESTS POUR GET_ALL_USERS (read_users)
# =========================================================

def test_read_users_success(mock_db, mock_current_admin, sample_users):
    """
    Test 1: Récupération réussie de tous les utilisateurs
    """
    # Arrange
    with patch('app.routers.admin_router.get_all_users') as mock_get_all_users:
        mock_get_all_users.return_value = sample_users
        
        # Act
        result = admin_router.routes[0].endpoint(
            db=mock_db,
            current_user=mock_current_admin
        )
        
        # Assert
        assert len(result) == 2
        assert result[0].email == "admin@example.com"
        assert result[1].email == "user@example.com"
        mock_get_all_users.assert_called_once_with(db=mock_db)


def test_read_users_exception(mock_db, mock_current_admin):
    """
    Test 2: Gestion d'erreur lors de la récupération des utilisateurs
    """
    # Arrange
    with patch('app.routers.admin_router.get_all_users') as mock_get_all_users:
        mock_get_all_users.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            admin_router.routes[0].endpoint(
                db=mock_db,
                current_user=mock_current_admin
            )
        
        assert exc_info.value.status_code == 500
        assert "Error fetching users" in str(exc_info.value.detail)


# =========================================================
# TESTS POUR GET_ALL_HISTORIES (ADMIN)
# =========================================================

def test_get_all_histories_success(mock_db, mock_current_admin):
    """
    Test 1: Récupération réussie des historiques
    """
    # Arrange
    expected_result = {
        "page": 1,
        "limit": 20,
        "total": 10,
        "data": []
    }
    
    with patch('app.routers.admin_router.get_all_histories_repo') as mock_get_histories:
        mock_get_histories.return_value = expected_result
        
        # Act
        result = admin_router.routes[1].endpoint(
            user_id=None,
            page=1,
            limit=20,
            variety=None,
            maturity=None,
            date_from=None,
            date_to=None,
            db=mock_db,
            current_user=mock_current_admin
        )
        
        # Assert
        assert result == expected_result
        mock_get_histories.assert_called_once_with(
            db=mock_db,
            user_id=None,
            page=1,
            limit=20,
            variety=None,
            maturity=None,
            date_from=None,
            date_to=None
        )


def test_get_all_histories_with_filters(mock_db, mock_current_admin):
    """
    Test 2: Récupération avec filtres
    """
    # Arrange
    user_id = 1
    page = 2
    limit = 10
    variety = "Majhoul"
    maturity = "Stage 3"
    date_from = datetime(2024, 1, 1)
    date_to = datetime(2024, 1, 31)
    
    expected_result = {
        "page": 2,
        "limit": 10,
        "total": 5,
        "data": []
    }
    
    with patch('app.routers.admin_router.get_all_histories_repo') as mock_get_histories:
        mock_get_histories.return_value = expected_result
        
        # Act
        result = admin_router.routes[1].endpoint(
            user_id=user_id,
            page=page,
            limit=limit,
            variety=variety,
            maturity=maturity,
            date_from=date_from,
            date_to=date_to,
            db=mock_db,
            current_user=mock_current_admin
        )
        
        # Assert
        assert result == expected_result
        mock_get_histories.assert_called_once_with(
            db=mock_db,
            user_id=user_id,
            page=page,
            limit=limit,
            variety=variety,
            maturity=maturity,
            date_from=date_from,
            date_to=date_to
        )


def test_get_all_histories_exception(mock_db, mock_current_admin):
    """
    Test 3: Gestion d'erreur lors de la récupération des historiques
    """
    # Arrange
    with patch('app.routers.admin_router.get_all_histories_repo') as mock_get_histories:
        mock_get_histories.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            admin_router.routes[1].endpoint(
                user_id=None,
                page=1,
                limit=20,
                variety=None,
                maturity=None,
                date_from=None,
                date_to=None,
                db=mock_db,
                current_user=mock_current_admin
            )
        
        assert exc_info.value.status_code == 500
        assert "Error fetching histories" in str(exc_info.value.detail)


# =========================================================
# TESTS POUR GET_USER_STATS_ADMIN
# =========================================================

@pytest.mark.anyio
async def test_get_user_stats_admin_success(mock_db, mock_current_admin):
    """
    Test 1: Récupération réussie des stats utilisateur
    """
    # Arrange
    user_id = 2
    expected_stats = {
        "user": {"id": 2, "username": "testuser", "email": "test@example.com"},
        "summary": {"total_predictions": 10},
        "classifications": {"varieties": [], "maturities": []}
    }
    
    with patch('app.routers.admin_router.get_user_statistics') as mock_get_stats:
        mock_get_stats.return_value = expected_stats
        
        # Act
        result = await admin_router.routes[2].endpoint(
            user_id=user_id,
            current_user=mock_current_admin,
            db=mock_db
        )
        
        # Assert
        assert result == expected_stats
        mock_get_stats.assert_called_once_with(
            db=mock_db,
            user_id=user_id,
            requesting_user_id=mock_current_admin.id,
            requesting_user_role=mock_current_admin.role
        )


@pytest.mark.anyio
async def test_get_user_stats_admin_not_found(mock_db, mock_current_admin):
    """
    Test 2: Utilisateur non trouvé
    """
    # Arrange
    user_id = 999
    
    with patch('app.routers.admin_router.get_user_statistics') as mock_get_stats:
        mock_get_stats.side_effect = ValueError("Utilisateur non trouvé")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await admin_router.routes[2].endpoint(
                user_id=user_id,
                current_user=mock_current_admin,
                db=mock_db
            )
        
        assert exc_info.value.status_code == 404
        assert "Utilisateur non trouvé" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_get_user_stats_admin_unauthorized(mock_db, mock_current_admin):
    """
    Test 3: Erreur de permission
    """
    # Arrange
    user_id = 3
    
    with patch('app.routers.admin_router.get_user_statistics') as mock_get_stats:
        mock_get_stats.side_effect = PermissionError("Vous n'avez pas les droits")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await admin_router.routes[2].endpoint(
                user_id=user_id,
                current_user=mock_current_admin,
                db=mock_db
            )
        
        assert exc_info.value.status_code == 500
        assert "Erreur lors de la récupération" in str(exc_info.value.detail)


# =========================================================
# TESTS POUR GET_ADMIN_STATS
# =========================================================

def test_get_admin_stats_success(mock_db, mock_current_admin):
    """
    Test 1: Récupération réussie des statistiques admin
    """
    # Arrange
    expected_stats = {
        "total_users": 150,
        "active_users_count": 45,
        "total_queries": 1200,
        "total_detections": 3500,
        "top_varieties": [{"name": "Majhoul", "count": 500}],
        "top_maturities": [{"name": "Stage 3", "count": 800}]
    }
    
    with patch('app.routers.admin_router.get_admin_stats_repo') as mock_get_stats:
        mock_get_stats.return_value = expected_stats
        
        # Act
        result = admin_router.routes[3].endpoint(
            db=mock_db,
            current_user=mock_current_admin
        )
        
        # Assert
        assert result == expected_stats
        mock_get_stats.assert_called_once_with(mock_db)


def test_get_admin_stats_exception(mock_db, mock_current_admin):
    """
    Test 2: Gestion d'erreur lors de la récupération des stats
    """
    # Arrange
    with patch('app.routers.admin_router.get_admin_stats_repo') as mock_get_stats:
        mock_get_stats.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            admin_router.routes[3].endpoint(
                db=mock_db,
                current_user=mock_current_admin
            )
        
        assert exc_info.value.status_code == 500
        assert "Error fetching stats" in str(exc_info.value.detail)


def test_get_admin_stats_empty(mock_db, mock_current_admin):
    """
    Test 3: Statistiques vides
    """
    # Arrange
    expected_stats = {
        "total_users": 0,
        "active_users_count": 0,
        "total_queries": 0,
        "total_detections": 0,
        "top_varieties": [],
        "top_maturities": []
    }
    
    with patch('app.routers.admin_router.get_admin_stats_repo') as mock_get_stats:
        mock_get_stats.return_value = expected_stats
        
        # Act
        result = admin_router.routes[3].endpoint(
            db=mock_db,
            current_user=mock_current_admin
        )
        
        # Assert
        assert result == expected_stats
        assert result["total_users"] == 0
        assert result["top_varieties"] == []