from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Local imports
from app.core.config import settings
from app.core.logger import logger

# =========================================================
# 🟢 ENGINE CONFIGURATION
# =========================================================

engine = create_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True
)

Base = declarative_base()


# =========================================================
# 🟢 DATABASE UTILS
# =========================================================

def get_db():
    """Générateur de session SQLAlchemy pour injection de dépendances FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialise la base de données en créant toutes les tables définies."""
    from app import models
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"☑️ Tables de la base de données créées avec succès !")
    except Exception as e:
        logger.exception(f"❌ Erreur lors de l'initialisation de la DB : {e}")