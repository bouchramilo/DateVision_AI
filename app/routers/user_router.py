from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_
from typing import Optional
from datetime import datetime

from app.core.database import get_db
from app.models.history_model import History
from app.models.result_model import Result
# app/api/routes/history.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
from app.core.database import get_db
from app.services.history_service import get_user_statistics
from app.core.deps import get_current_user
from app.models.user_model import User

router = APIRouter(prefix="/history", tags=["history"])



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
user_router = APIRouter(prefix="/users", tags=["users"])


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@user_router.get("/history/{user_id}")
def get_user_history(
    user_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, le=100),

    variety: Optional[str] = None,
    maturity: Optional[str] = None,

    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,

    db: Session = Depends(get_db)
):

    query = db.query(History).options(joinedload(History.results))

    # filtre user
    query = query.filter(History.user_id == user_id)

    # filtre date
    if date_from and date_to:
        query = query.filter(
            History.created_at.between(date_from, date_to)
        )

    # filtre variété / maturité (via join)
    if variety or maturity:
        query = query.join(Result)

        if variety:
            query = query.filter(Result.variety == variety)

        if maturity:
            query = query.filter(Result.maturity == maturity)

    # total count
    total = query.count()

    # pagination
    histories = query.order_by(History.created_at.desc()) \
                     .offset((page - 1) * limit) \
                     .limit(limit) \
                     .all()

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "data": histories
    }
    
    
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


@user_router.get("/my-stats", response_model=Dict[str, Any])
async def get_my_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les statistiques de l'utilisateur connecté
    """
    try:
        stats = get_user_statistics(
            db=db,
            user_id=current_user.id,
            requesting_user_id=current_user.id,
            requesting_user_role=current_user.role
        )
        return stats
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
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