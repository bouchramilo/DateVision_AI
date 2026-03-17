from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings
from app.core.logger import logger


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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

def init_db():
    from app import models
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"☑️☑️ Tables créées avec succès ! ☑️")
    except Exception as e:
        logger.exception(f"❌❌ Erreur lors de la création des tables : {e} ❌")
    
    