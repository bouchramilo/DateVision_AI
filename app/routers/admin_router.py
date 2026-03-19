from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.schemas.user_schema import UserCreate, UserLogin, UserLoginResponse, User
from app.core.database import get_db
from app.repositories.user_repository import get_all_users
from app.core.logger import logger
from app.core.deps import get_current_admin_user

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
admin_router = APIRouter(prefix="/admin", tags=["admin routes"])

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ALL USERS
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@admin_router.get("/all-users", response_model=list[User], status_code=status.HTTP_200_OK)
def read_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    try:
        users = get_all_users(db=db)
        return users
    except Exception as e:
        logger.error(f"❌❌ Erreur lors de la récupération des utilisateurs : {e} ❌")
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la récupération des utilisateurs"
        )