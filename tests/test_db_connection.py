# tests/test_db_connection.py
import pytest 
from sqlalchemy import text
from app.core.database import engine, SessionLocal, get_db

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def test_database_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
            print("\n✅ Connecxion Database: OK")
    except Exception as e:
        pytest.fail(f"❌ Échec de la connexion à la base de données: {str(e)}")

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def test_session_creation():
    db = SessionLocal()
    assert db is not None
    db.close()
    print("Création Session (SessionLocal): OK")
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def test_get_db_dependency():
    getdb = get_db()
    db = next(getdb)
    assert db is not None
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::