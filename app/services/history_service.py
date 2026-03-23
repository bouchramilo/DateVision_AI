# app/services/history_service.py
from sqlalchemy.orm import Session
from typing import Dict, Any
from app.repositories.history_repository import get_user_stats
from app.models.user_model import User


def get_user_statistics(
    db: Session,
    user_id: int,
    requesting_user_id: int,
    requesting_user_role: str
) -> Dict[str, Any]:
    """
    Récupère les statistiques d'un utilisateur
    - Les utilisateurs normaux ne peuvent voir que leurs propres statistiques
    - Les admins peuvent voir les statistiques de n'importe quel utilisateur
    """
    # Vérifier les permissions
    if requesting_user_role != 'ADMIN' and requesting_user_id != user_id:
        raise PermissionError("Vous n'avez pas les droits pour voir ces statistiques")
    
    # Vérifier que l'utilisateur existe
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("Utilisateur non trouvé")
    
    # Récupérer les statistiques
    stats = get_user_stats(db, user_id)
    
    return stats