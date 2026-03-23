from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime
from app.core.database import get_db
from app.core.logger import logger
from app.core.deps import get_current_admin_user
from app.repositories.admin_repository import get_admin_stats_repo
from app.schemas.user_schema import User as UserSchema
from app.repositories.user_repository import get_all_users
from app.repositories.history_repository import get_all_histories_repo
from app.services.history_service import get_user_statistics
from app.models.user_model import User as UserModel


admin_router = APIRouter(prefix="/admin", tags=["admin routes"])


# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 🔹 ALL USERS
# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@admin_router.get("/all-users", response_model=list[UserSchema])
def read_users(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_admin_user)
):
    try:
        return get_all_users(db=db)
    except Exception as e:
        logger.error(f"❌ Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Error fetching users")


# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 🔹 ALL HISTORIES (ADMIN)
# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@admin_router.get("/history")
def get_all_histories(
    user_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, le=100),
    variety: Optional[str] = None,
    maturity: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_admin_user)
):
    try:
        return get_all_histories_repo(
            db=db,
            user_id=user_id,
            page=page,
            limit=limit,
            variety=variety,
            maturity=maturity,
            date_from=date_from,
            date_to=date_to
        )
    except Exception as e:
        logger.error(f"❌ Error fetching histories: {e}")
        raise HTTPException(status_code=500, detail="Error fetching histories")
    
    
# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 🔹 ALL HISTORIES (ADMIN)
# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


@admin_router.get("/users/{user_id}/stats", response_model=Dict[str, Any])
async def get_user_stats_admin(
    user_id: int,
    current_user: UserModel = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les statistiques détaillées d'un utilisateur spécifique (admin uniquement)
    """
    try:
        stats = get_user_statistics(
            db=db,
            user_id=user_id,
            requesting_user_id=current_user.id,
            requesting_user_role=current_user.role
        )
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )
        
        
        
        


# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 🔹 ALL STATS (ADMIN)
# !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@admin_router.get("/stats")
def get_admin_stats(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_admin_user)
):
    try:
        return get_admin_stats_repo(db)
    except Exception as e:
        logger.error(f"❌ Error fetching admin stats: {e}")
        raise HTTPException(status_code=500, detail="Error fetching stats")