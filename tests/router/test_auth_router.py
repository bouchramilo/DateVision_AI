"""
Tests unitaires pour les routes d'authentification (version simplifiée)
"""

import pytest
from fastapi import HTTPException
from unittest.mock import Mock, patch
from datetime import datetime
from unittest.mock import Mock, patch

from app.routers.auth_router import auth_router
from app.schemas.user_schema import UserCreate


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock()


@pytest.fixture
def mock_user():
    """Mock d'un utilisateur"""
    user = Mock()
    user.id = 1
    user.email = "test@example.com"
    user.username = "testuser"
    user.password_hash = "hashed_password"
    user.role = "USER"
    user.is_active = False
    user.created_at = datetime.utcnow()
    return user


# =========================================================
# TESTS POUR REGISTER
# =========================================================

@pytest.mark.anyio
async def test_register_success(mock_db, mock_user):
    """Test: Inscription réussie"""
    data = {
        "email": "new@example.com",
        "username": "newuser",
        "password": "password123",
        "password_repeat": "password123",
        "role": "USER"
    }
    print(f"DEBUG: Input data = {data}")
    user_data = UserCreate(**data)
    
    with patch('app.routers.auth_router.check_user_existe', return_value=None):
        with patch('app.routers.auth_router.create_user', return_value=mock_user):
            print(f"DEBUG: user_data = {user_data.model_dump()}")
            result = await auth_router.routes[0].endpoint(
                user_data=user_data,
                db=mock_db
            )
            
            assert result.id == mock_user.id
            assert result.email == mock_user.email


@pytest.mark.anyio
async def test_register_user_exists(mock_db, mock_user):
    """Test: Inscription avec email existant"""
    user_data = UserCreate(
        email="existing@example.com",
        username="existing",
        password="password",
        password_repeat="password",
        role="USER"
    )
    
    with patch('app.routers.auth_router.check_user_existe', return_value=mock_user):
        with pytest.raises(HTTPException) as exc:
            await auth_router.routes[0].endpoint(
                user_data=user_data,
                db=mock_db
            )
        
        assert exc.value.status_code == 400
        assert "User déjà existant" in str(exc.value.detail)


@pytest.mark.anyio
async def test_register_error(mock_db):
    """Test: Erreur lors de l'inscription"""
    user_data = UserCreate(
        email="test@example.com",
        username="test",
        password="password",
        password_repeat="password",
        role="USER"
    )
    
    with patch('app.routers.auth_router.check_user_existe', side_effect=Exception("DB error")):
        with pytest.raises(HTTPException) as exc:
            await auth_router.routes[0].endpoint(
                user_data=user_data,
                db=mock_db
            )
        
        assert exc.value.status_code == 500


# =========================================================
# TESTS POUR LOGIN
# =========================================================

def test_login_success(mock_db, mock_user):
    """Test: Connexion réussie"""
    form_data = Mock()
    form_data.username = "test@example.com"
    form_data.password = "password123"
    
    with patch('app.routers.auth_router.get_user_by_email', return_value=mock_user):
        with patch('app.routers.auth_router.verify_password', return_value=True):
            with patch('app.routers.auth_router.update_activation_user', return_value=mock_user):
                with patch('app.routers.auth_router.create_access_token', return_value="mock_token"):
                    result = auth_router.routes[1].endpoint(
                        form_data=form_data,
                        db=mock_db
                    )
                    
                    assert result["access_token"] == "mock_token"
                    assert result["token_type"] == "bearer"


def test_login_user_not_found(mock_db):
    """Test: Connexion avec utilisateur inexistant"""
    form_data = Mock()
    form_data.username = "nonexistent@example.com"
    form_data.password = "password"
    
    with patch('app.routers.auth_router.get_user_by_email', return_value=None):
        with pytest.raises(HTTPException) as exc:
            auth_router.routes[1].endpoint(
                form_data=form_data,
                db=mock_db
            )
        
        assert exc.value.status_code == 401
        assert "Incorrect username or password" in str(exc.value.detail)


def test_login_wrong_password(mock_db, mock_user):
    """Test: Connexion avec mot de passe incorrect"""
    form_data = Mock()
    form_data.username = "test@example.com"
    form_data.password = "wrongpassword"
    
    with patch('app.routers.auth_router.get_user_by_email', return_value=mock_user):
        with patch('app.routers.auth_router.verify_password', return_value=False):
            with pytest.raises(HTTPException) as exc:
                auth_router.routes[1].endpoint(
                    form_data=form_data,
                    db=mock_db
                )
            
            assert exc.value.status_code == 401


# =========================================================
# TESTS POUR GET_ME
# =========================================================

def test_get_me_success():
    """Test: Récupération de l'utilisateur connecté"""
    mock_current_user = Mock()
    mock_current_user.id = 1
    mock_current_user.email = "test@example.com"
    
    result = auth_router.routes[2].endpoint(
        current_user=mock_current_user
    )
    
    assert result == mock_current_user


def test_get_me_returns_user_data():
    """Test: Vérification des données utilisateur"""
    mock_current_user = Mock()
    mock_current_user.id = 5
    mock_current_user.email = "user@example.com"
    mock_current_user.username = "username"
    
    result = auth_router.routes[2].endpoint(
        current_user=mock_current_user
    )
    
    assert result.id == 5
    assert result.email == "user@example.com"
    assert result.username == "username"