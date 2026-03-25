"""
Tests unitaires pour les utilitaires de hashage de mot de passe
"""

import pytest
import bcrypt

from app.security.hashing import _prehash, hash_password, verify_password


# =========================================================
# TESTS POUR _PREHASH
# =========================================================

def test_prehash_returns_bytes():
    """Test: _prehash retourne des bytes"""
    result = _prehash("password")
    assert isinstance(result, bytes)
    assert len(result) == 32  # SHA-256 = 32 bytes


def test_prehash_same_password_same_result():
    """Test: Même mot de passe produit le même hash"""
    password = "test123"
    assert _prehash(password) == _prehash(password)


def test_prehash_different_password_different_result():
    """Test: Différents mots de passe produisent des hash différents"""
    assert _prehash("pass1") != _prehash("pass2")


# =========================================================
# TESTS POUR HASH_PASSWORD
# =========================================================

def test_hash_password_returns_bcrypt_string():
    """Test: hash_password retourne une string au format bcrypt"""
    result = hash_password("password")
    assert isinstance(result, str)
    assert result.startswith('$2b$')
    assert len(result) == 60


def test_hash_password_same_password_different_hash():
    """Test: Même mot de passe produit des hash différents (salt aléatoire)"""
    hash1 = hash_password("password")
    hash2 = hash_password("password")
    assert hash1 != hash2


def test_hash_password_handles_long_password():
    """Test: Gère les mots de passe longs (>72 caractères)"""
    long_password = "a" * 100
    result = hash_password(long_password)
    assert isinstance(result, str)
    assert result.startswith('$2b$')


# =========================================================
# TESTS POUR VERIFY_PASSWORD
# =========================================================

def test_verify_password_correct():
    """Test: Vérification réussie avec le bon mot de passe"""
    password = "correct_password"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True


def test_verify_password_incorrect():
    """Test: Vérification échoue avec le mauvais mot de passe"""
    password = "correct_password"
    hashed = hash_password(password)
    assert verify_password("wrong_password", hashed) is False


def test_verify_password_long_password():
    """Test: Vérification avec mot de passe long (>72 caractères)"""
    long_password = "a" * 100
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed) is True