# app/repositories/admin_repository.py

from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from app.models.user_model import User
from app.models.history_model import History
from app.models.result_model import Result


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_admin_stats_repo(db: Session):
    # total users
    total_users = db.query(func.count(User.id)).scalar()

    # active users 
    last_7_days = datetime.utcnow() - timedelta(days=30)

    active_users = (
        db.query(func.count(func.distinct(History.user_id)))
        .filter(History.created_at >= last_7_days)
        .scalar()
    )

    # total analyses
    total_queries = db.query(func.count(History.id)).scalar()

    # total detections
    total_detections = db.query(func.count(Result.id)).scalar()

    # top varieties
    top_varieties = (
        db.query(
            Result.variety,
            func.count(Result.id).label("count")
        )
        .group_by(Result.variety)
        .order_by(func.count(Result.id).desc())
        .limit(5)
        .all()
    )

    # top maturities
    top_maturities = (
        db.query(
            Result.maturity,
            func.count(Result.id).label("count")
        )
        .group_by(Result.maturity)
        .order_by(func.count(Result.id).desc())
        .limit(5)
        .all()
    )

    return {
        "total_users": total_users,
        "active_users_count": active_users,
        "total_queries": total_queries,
        "total_detections": total_detections,

        "top_varieties": [
            {"name": v.variety, "count": v.count} for v in top_varieties
        ],

        "top_maturities": [
            {"name": m.maturity, "count": m.count} for m in top_maturities
        ],
    }