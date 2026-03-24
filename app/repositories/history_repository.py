# app/repositories/history_repository.py
# app/repositories/history_repository.py
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import Dict, List, Any
from app.models.history_model import History
from app.models.result_model import Result
from app.models.user_model import User
from sqlalchemy.orm import Session, joinedload
from typing import Optional
from datetime import datetime
from app.models.history_model import History
from app.models.result_model import Result


# !=========================================================
# 🔹 CREATE (SAVE)
# !=========================================================
def save_full_result(db, user_id, image, result):

    # 1. créer history
    history = History(
        user_id=user_id,
        image=image,
        report=result.get("report")
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    # 2. ajouter detections
    for det in result.get("detections", []):
        detection = Result(
            history_id=history.id,

            x1=det["bbox"][0],
            y1=det["bbox"][1],
            x2=det["bbox"][2],
            y2=det["bbox"][3],

            detection_score=det["detection_score"],

            variety=det["variety"],
            variety_score=det["variety_score"],

            maturity=det["maturity"],
            maturity_score=det["maturity_score"]
        )
        db.add(detection)

    db.commit()

    return history


# !=========================================================
# 🔹 MAIN FUNCTION (READ)
# !=========================================================
def get_all_histories_repo(
    db: Session,
    user_id: Optional[int] = None,
    page: int = 1,
    limit: int = 20,
    variety: Optional[str] = None,
    maturity: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
):
    query = db.query(History).options(
        joinedload(History.results),
        joinedload(History.user)
    )

    #  filtre user
    if user_id:
        query = query.filter(History.user_id == user_id)

    #  filtre date
    if date_from and date_to:
        query = query.filter(History.created_at.between(date_from, date_to))

    #  filtre résultats
    if variety or maturity:
        query = query.join(Result)

        if variety:
            query = query.filter(Result.variety == variety)

        if maturity:
            query = query.filter(Result.maturity == maturity)

    #  total
    total = query.count()

    #  pagination
    histories = (
        query.order_by(History.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "data": histories
    }
    
    
    



# !=========================================================
# GET STATS (par USER)
# !=========================================================
def get_user_stats(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère les statistiques complètes d'un utilisateur"""
    
    # Statistiques de base
    summary = get_user_summary_stats(db, user_id)
    
    # Classifications par variété et maturité
    classifications = get_user_classifications_stats(db, user_id)
    
    # Activité hebdomadaire
    weekly_activity = get_user_weekly_activity(db, user_id)
    
    # Distribution des scores de détection
    detection_scores = get_user_detection_scores_distribution(db, user_id)
    
    # Prédictions récentes
    recent_predictions = get_user_recent_predictions(db, user_id)
    
    # Informations utilisateur
    user_info = db.query(User).filter(User.id == user_id).first()
    
    return {
        "user": {
            "id": user_info.id,
            "username": user_info.username,
            "email": user_info.email,
            "created_at": user_info.created_at
        },
        "summary": summary,
        "classifications": classifications,
        "weekly_activity": weekly_activity,
        "detection_scores": detection_scores,
        "recent_predictions": recent_predictions
    }



# !=========================================================
# GET STATS (par ADMIN)
# !=========================================================
def get_user_summary_stats(db: Session, user_id: int) -> Dict[str, Any]:
    """Statistiques résumées de l'utilisateur"""
    
    # Nombre total d'images traitées
    total_images = db.query(func.count(History.id)).filter(
        History.user_id == user_id
    ).scalar() or 0
    
    # Nombre total de détections
    total_detections = db.query(func.count(Result.id)).join(
        History, Result.history_id == History.id
    ).filter(
        History.user_id == user_id
    ).scalar() or 0
    
    # Score de détection moyen
    avg_detection_score = db.query(func.avg(Result.detection_score)).join(
        History, Result.history_id == History.id
    ).filter(
        History.user_id == user_id
    ).scalar() or 0.0
    
    # Nombre de variétés uniques
    unique_varieties = db.query(func.count(func.distinct(Result.variety))).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.variety.isnot(None)
        )
    ).scalar() or 0
    
    # Nombre de maturités uniques
    unique_maturities = db.query(func.count(func.distinct(Result.maturity))).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.maturity.isnot(None)
        )
    ).scalar() or 0
    
    return {
        "total_images_processed": total_images,
        "total_detections": total_detections,
        "total_predictions": total_detections,
        "average_detection_score": float(avg_detection_score),
        "unique_varieties": unique_varieties,
        "unique_maturities": unique_maturities
    }


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_user_classifications_stats(db: Session, user_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """Statistiques par variété et maturité"""
    
    # Statistiques par variété
    variety_stats = db.query(
        Result.variety,
        func.count(Result.id).label('count'),
        func.avg(Result.variety_score).label('avg_score')
    ).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.variety.isnot(None)
        )
    ).group_by(Result.variety).all()
    
    varieties = []
    for v in variety_stats:
        varieties.append({
            "variety": v.variety,
            "count": v.count,
            "avg_score": float(v.avg_score) if v.avg_score else 0.0
        })
    
    # Statistiques par maturité
    maturity_stats = db.query(
        Result.maturity,
        func.count(Result.id).label('count'),
        func.avg(Result.maturity_score).label('avg_score')
    ).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.maturity.isnot(None)
        )
    ).group_by(Result.maturity).all()
    
    maturities = []
    for m in maturity_stats:
        maturities.append({
            "maturity": m.maturity,
            "count": m.count,
            "avg_score": float(m.avg_score) if m.avg_score else 0.0
        })
    
    return {
        "varieties": varieties,
        "maturities": maturities
    }

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_user_weekly_activity(db: Session, user_id: int) -> List[Dict[str, Any]]:
    """Activité des 7 derniers jours"""
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Requête pour compter les prédictions par jour
    daily_activity = db.query(
        func.date(History.created_at).label('date'),
        func.count(Result.id).label('count')
    ).join(
        Result, History.id == Result.history_id
    ).filter(
        and_(
            History.user_id == user_id,
            History.created_at >= start_date,
            History.created_at <= end_date
        )
    ).group_by(
        func.date(History.created_at)
    ).order_by(
        func.date(History.created_at)
    ).all()
    
    # Remplir les jours manquants avec 0
    activity_dict = {}
    for day in daily_activity:
        activity_dict[str(day.date)] = day.count
    
    result = []
    for i in range(7):
        date = (end_date - timedelta(days=6-i)).date()
        result.append({
            "date": date.isoformat(),
            "count": activity_dict.get(str(date), 0)
        })
    
    return result


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_user_detection_scores_distribution(db: Session, user_id: int) -> Dict[str, int]:
    """Distribution des scores de détection"""
    
    # Excellent (> 0.8)
    excellent = db.query(func.count(Result.id)).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.detection_score > 0.8
        )
    ).scalar() or 0
    
    # Bon (0.6 - 0.8)
    good = db.query(func.count(Result.id)).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.detection_score >= 0.6,
            Result.detection_score <= 0.8
        )
    ).scalar() or 0
    
    # Moyen (0.4 - 0.6)
    fair = db.query(func.count(Result.id)).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.detection_score >= 0.4,
            Result.detection_score < 0.6
        )
    ).scalar() or 0
    
    # Faible (< 0.4)
    poor = db.query(func.count(Result.id)).join(
        History, Result.history_id == History.id
    ).filter(
        and_(
            History.user_id == user_id,
            Result.detection_score < 0.4
        )
    ).scalar() or 0
    
    return {
        "excellent": excellent,
        "good": good,
        "fair": fair,
        "poor": poor
    }

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_user_recent_predictions(db: Session, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Récupère les prédictions récentes de l'utilisateur"""
    
    recent = db.query(
        History.id,
        History.image,
        History.created_at,
        func.count(Result.id).label('detection_count'),
        func.avg(Result.detection_score).label('avg_score')
    ).outerjoin(
        Result, History.id == Result.history_id
    ).filter(
        History.user_id == user_id
    ).group_by(
        History.id
    ).order_by(
        History.created_at.desc()
    ).limit(limit).all()
    
    result = []
    for r in recent:
        result.append({
            "id": r.id,
            "image": r.image,
            "created_at": r.created_at,
            "detection_count": r.detection_count,
            "avg_detection_score": float(r.avg_score) if r.avg_score else 0.0
        })
    
    return result