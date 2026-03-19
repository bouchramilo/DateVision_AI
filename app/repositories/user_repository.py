from app.models.user_model import User as User_Model
from app.schemas.user_schema import UserCreate, UserUpdate
from app.security.hashing import hash_password
from sqlalchemy.orm import Session
from app.core.logger import logger
from typing import Union

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def check_user_existe(db: Session, email: str):
    return (
        db.query(User_Model)
        .filter(User_Model.email == email)
        .first()
    )

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def create_user(db: Session, user: UserCreate):
    db_user = User_Model(
        email=user.email,
        username=user.username,
        password_hash=hash_password(user.password),
        role=user.role,
        is_active=False
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_all_users(db: Session):
    return db.query(User_Model).all()


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_user_by_email(db: Session, email: str):
    return db.query(User_Model).filter(User_Model.email == email).first()


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def update_activation_user(db: Session, user: Union[UserUpdate, User_Model], is_active: bool = True):

    if isinstance(user, User_Model):
        db_user = user
    else:
        db_user = (
            db.query(User_Model)
            .filter(
                User_Model.email == user.email
            )
            .first()
        )

    if not db_user:
        logger.warning(f"🟡🟡 User not found for activation update: {user}")
        return None
    
    db_user.is_active = is_active

    db.commit()
    db.refresh(db_user)
    
    return db_user
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

