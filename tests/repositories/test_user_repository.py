"""
Tests unitaires pour le repository utilisateur
"""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.repositories.user_repository import (
    check_user_existe,
    create_user,
    get_all_users,
    get_user_by_email,
    update_activation_user
)
from app.models.user_model import User as User_Model
from app.schemas.user_schema import UserCreate, UserUpdate


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
    user = Mock(spec=User_Model)
    user.id = 1
    user.email = "test@example.com"
    user.username = "testuser"
    user.role = "USER"
    user.is_active = False
    return user


@pytest.fixture
def user_create_data():
    """Données pour créer un utilisateur"""
    return UserCreate(
        email="newuser@example.com",
        username="newuser",
        password="password123",
        password_repeat="password123",
        role="USER"
    )


# =========================================================
# TESTS POUR CHECK_USER_EXISTE
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_check_user_existe_found(mock_db, mock_user):
    """Test: Utilisateur trouvé par email"""
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    result = check_user_existe(mock_db, "test@example.com")
    assert result == mock_user


# --------------------------------------------------------------------------------------------------------
def test_check_user_existe_not_found(mock_db):
    """Test: Utilisateur non trouvé"""
    mock_db.query.return_value.filter.return_value.first.return_value = None
    result = check_user_existe(mock_db, "nonexistent@example.com")
    assert result is None


# =========================================================
# TESTS POUR CREATE_USER
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_create_user_success(mock_db, user_create_data):
    """Test: Création réussie d'un utilisateur"""
    with patch('app.repositories.user_repository.hash_password') as mock_hash:
        mock_hash.return_value = "hashed_password"
        
        result = create_user(mock_db, user_create_data)
        
        assert result is not None
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()


# --------------------------------------------------------------------------------------------------------
def test_create_user_fields_set(mock_db, user_create_data):
    """Test: Vérification que tous les champs sont correctement définis"""
    with patch('app.repositories.user_repository.hash_password') as mock_hash:
        mock_hash.return_value = "hashed_password"
        
        create_user(mock_db, user_create_data)
        
        user_arg = mock_db.add.call_args[0][0]
        assert user_arg.email == user_create_data.email
        assert user_arg.username == user_create_data.username
        assert user_arg.password_hash == "hashed_password"
        assert user_arg.role == user_create_data.role
        assert user_arg.is_active is False


# =========================================================
# TESTS POUR GET_ALL_USERS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_all_users_success(mock_db):
    """Test: Récupération de tous les utilisateurs"""
    mock_users = [Mock(), Mock()]
    mock_db.query.return_value.all.return_value = mock_users
    
    result = get_all_users(mock_db)
    
    assert len(result) == 2
    assert result == mock_users


# --------------------------------------------------------------------------------------------------------
def test_get_all_users_empty(mock_db):
    """Test: Aucun utilisateur"""
    mock_db.query.return_value.all.return_value = []
    
    result = get_all_users(mock_db)
    
    assert result == []


# =========================================================
# TESTS POUR GET_USER_BY_EMAIL
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_by_email_found(mock_db, mock_user):
    """Test: Utilisateur trouvé par email"""
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    result = get_user_by_email(mock_db, "test@example.com")
    
    assert result == mock_user


# --------------------------------------------------------------------------------------------------------
def test_get_user_by_email_not_found(mock_db):
    """Test: Utilisateur non trouvé"""
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    result = get_user_by_email(mock_db, "nonexistent@example.com")
    
    assert result is None


# =========================================================
# TESTS POUR UPDATE_ACTIVATION_USER
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_update_activation_with_model(mock_db, mock_user):
    """Test: Activation avec objet User_Model"""
    result = update_activation_user(mock_db, mock_user, is_active=True)
    
    assert result == mock_user
    assert mock_user.is_active is True
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()


# --------------------------------------------------------------------------------------------------------
def test_update_activation_with_schema(mock_db, mock_user):
    """Test: Activation avec objet UserUpdate"""
    user_update = UserUpdate(email="test@example.com")
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    result = update_activation_user(mock_db, user_update, is_active=True)
    
    assert result == mock_user
    assert mock_user.is_active is True


# --------------------------------------------------------------------------------------------------------
def test_update_activation_user_not_found(mock_db):
    """Test: Utilisateur non trouvé"""
    user_update = UserUpdate(email="nonexistent@example.com")
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    result = update_activation_user(mock_db, user_update, is_active=True)
    
    assert result is None
    mock_db.commit.assert_not_called()
