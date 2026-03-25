"""
Tests unitaires pour le repository d'historique
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.repositories.history_repository import (
    save_full_result,
    get_all_histories_repo,
    get_user_stats,
    get_user_summary_stats,
    get_user_classifications_stats,
    get_user_weekly_activity,
    get_user_detection_scores_distribution,
    get_user_recent_predictions
)
from app.models.history_model import History
from app.models.result_model import Result
from app.models.user_model import User


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock(spec=Session)


@pytest.fixture
def sample_result():
    """Exemple de résultat de détection"""
    return {
        "report": "Rapport de test",
        "detections": [
            {
                "bbox": [100, 150, 200, 250],
                "detection_score": 0.95,
                "variety": "Majhoul",
                "variety_score": 0.92,
                "maturity": "Stage 3",
                "maturity_score": 0.88
            },
            {
                "bbox": [300, 350, 400, 450],
                "detection_score": 0.87,
                "variety": "Boufagous",
                "variety_score": 0.85,
                "maturity": "Stage 2",
                "maturity_score": 0.82
            }
        ]
    }


# =========================================================
# TESTS POUR SAVE_FULL_RESULT
# =========================================================

# --------------------------------------------------------------------------------------------------------
@patch('app.repositories.history_repository.History')
@patch('app.repositories.history_repository.Result')
def test_save_full_result_success(mock_result_class, mock_history_class, mock_db, sample_result):
    """
    Test 1: Sauvegarde réussie d'un résultat complet
    """
    # Arrange
    user_id = 1
    image = b"fake_image_data"
    
    mock_history = mock_history_class.return_value
    mock_history.id = 1
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    mock_db.refresh.return_value = None
    
    # Act
    result = save_full_result(mock_db, user_id, image, sample_result)
    
    # Assert
    assert result == mock_history
    assert mock_db.commit.call_count == 2


# --------------------------------------------------------------------------------------------------------
@patch('app.repositories.history_repository.History')
def test_save_full_result_empty_detections(mock_history_class, mock_db):
    """
    Test 2: Sauvegarde avec aucune détection
    """
    # Arrange
    user_id = 1
    image = b"fake_image_data"
    empty_result = {"report": "Rapport", "detections": []}
    
    mock_history = mock_history_class.return_value
    mock_history.id = 1
    mock_db.add.return_value = None
    
    # Act
    result = save_full_result(mock_db, user_id, image, empty_result)
    
    # Assert
    assert result == mock_history
    assert mock_db.commit.call_count == 2


# --------------------------------------------------------------------------------------------------------
@patch('app.repositories.history_repository.History')
def test_save_full_result_without_detections_key(mock_history_class, mock_db):
    """
    Test 3: Sauvegarde sans clé detections dans le résultat
    """
    # Arrange
    user_id = 1
    image = b"fake_image_data"
    result_without_detections = {"report": "Rapport"}
    
    mock_history = mock_history_class.return_value
    mock_history.id = 1
    mock_db.add.return_value = None
    
    # Act
    result = save_full_result(mock_db, user_id, image, result_without_detections)
    
    # Assert
    assert result == mock_history
    assert mock_db.commit.call_count == 2


# =========================================================
# TESTS POUR GET_ALL_HISTORIES_REPO
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_all_histories_repo_no_filters(mock_db):
    """
    Test 1: Récupération sans filtres
    """
    # Arrange
    mock_histories = [Mock(spec=History) for _ in range(3)]
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.count.return_value = 10
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_histories
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_all_histories_repo(mock_db, page=1, limit=5)
    
    # Assert
    assert result["page"] == 1
    assert result["limit"] == 5
    assert result["total"] == 10
    assert len(result["data"]) == 3


# --------------------------------------------------------------------------------------------------------
def test_get_all_histories_repo_with_filters(mock_db):
    """
    Test 2: Récupération avec filtres (user_id, variety, maturity)
    """
    # Arrange
    user_id = 1
    variety = "Majhoul"
    maturity = "Stage 3"
    
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.count.return_value = 5
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_all_histories_repo(
        mock_db,
        user_id=user_id,
        variety=variety,
        maturity=maturity,
        page=2,
        limit=10
    )
    
    # Assert
    assert result["page"] == 2
    assert result["limit"] == 10
    assert result["total"] == 5


# --------------------------------------------------------------------------------------------------------
def test_get_all_histories_repo_with_date_filters(mock_db):
    """
    Test 3: Récupération avec filtres de dates
    """
    # Arrange
    date_from = datetime(2024, 1, 1)
    date_to = datetime(2024, 1, 31)
    
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.count.return_value = 3
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_all_histories_repo(
        mock_db,
        date_from=date_from,
        date_to=date_to
    )
    
    # Assert
    assert result["total"] == 3
    assert "data" in result


# =========================================================
# TESTS POUR GET_USER_SUMMARY_STATS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_summary_stats_success(mock_db):
    """
    Test 1: Récupération réussie des statistiques résumées
    """
    # Arrange
    user_id = 1
    
    # Mocks pour les 5 appels à query()...scalar()
    mock_scalars = [MagicMock() for _ in range(5)]
    mock_values = [10, 25, 0.75, 3, 2]
    for m, v in zip(mock_scalars, mock_values):
        # Certains ont filter(), d'autres join().filter()
        m.filter.return_value.scalar.return_value = v
        m.join.return_value.filter.return_value.scalar.return_value = v
    
    mock_db.query.side_effect = mock_scalars
    
    # Act
    result = get_user_summary_stats(mock_db, user_id)
    
    # Assert
    assert result["total_images_processed"] == 10
    assert result["total_detections"] == 25
    assert result["total_predictions"] == 25
    assert result["average_detection_score"] == 0.75
    assert result["unique_varieties"] == 3
    assert result["unique_maturities"] == 2


# --------------------------------------------------------------------------------------------------------
def test_get_user_summary_stats_empty(mock_db):
    """
    Test 2: Utilisateur sans données
    """
    # Arrange
    user_id = 1
    
    # Mock des requêtes retournant None
    mock_scalar = MagicMock()
    mock_scalar.filter.return_value.scalar.return_value = None
    mock_scalar.join.return_value.filter.return_value.scalar.return_value = None
    mock_db.query.return_value = mock_scalar
    
    # Act
    result = get_user_summary_stats(mock_db, user_id)
    
    # Assert
    assert result["total_images_processed"] == 0
    assert result["total_detections"] == 0
    assert result["average_detection_score"] == 0.0
    assert result["unique_varieties"] == 0
    assert result["unique_maturities"] == 0


# --------------------------------------------------------------------------------------------------------
def test_get_user_summary_stats_with_real_values(mock_db):
    """
    Test 3: Vérification des valeurs réelles
    """
    # Arrange
    user_id = 1
    
    mock_scalars = [MagicMock() for _ in range(5)]
    mock_values = [5, 15, 0.85, 2, 4]
    for m, v in zip(mock_scalars, mock_values):
        m.filter.return_value.scalar.return_value = v
        m.join.return_value.filter.return_value.scalar.return_value = v
    mock_db.query.side_effect = mock_scalars
    
    # Act
    result = get_user_summary_stats(mock_db, user_id)
    
    # Assert
    assert isinstance(result, dict)
    assert result["total_images_processed"] == 5
    assert result["average_detection_score"] == 0.85


# =========================================================
# TESTS POUR GET_USER_CLASSIFICATIONS_STATS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_classifications_stats_success(mock_db):
    """
    Test 1: Récupération réussie des stats de classification
    """
    # Arrange
    user_id = 1
    
    # Mock variety stats
    mock_varieties = [
        Mock(variety="Majhoul", count=10, avg_score=0.92),
        Mock(variety="Boufagous", count=5, avg_score=0.85)
    ]
    
    # Mock maturity stats
    mock_maturities = [
        Mock(maturity="Stage 3", count=8, avg_score=0.88),
        Mock(maturity="Stage 2", count=7, avg_score=0.82)
    ]
    
    mock_variety_query = MagicMock()
    mock_variety_query.join.return_value.filter.return_value.group_by.return_value.all.return_value = mock_varieties
    
    mock_maturity_query = MagicMock()
    mock_maturity_query.join.return_value.filter.return_value.group_by.return_value.all.return_value = mock_maturities
    
    mock_db.query.side_effect = [mock_variety_query, mock_maturity_query]
    
    # Act
    result = get_user_classifications_stats(mock_db, user_id)
    
    # Assert
    assert len(result["varieties"]) == 2
    assert len(result["maturities"]) == 2
    assert result["varieties"][0]["variety"] == "Majhoul"
    assert result["maturities"][0]["maturity"] == "Stage 3"


# --------------------------------------------------------------------------------------------------------
def test_get_user_classifications_stats_empty(mock_db):
    """
    Test 2: Utilisateur sans classifications
    """
    # Arrange
    user_id = 1
    
    mock_variety_query = MagicMock()
    mock_variety_query.join.return_value.filter.return_value.group_by.return_value.all.return_value = []
    mock_maturity_query = MagicMock()
    mock_maturity_query.join.return_value.filter.return_value.group_by.return_value.all.return_value = []
    mock_db.query.return_value = mock_variety_query
    
    # Act
    result = get_user_classifications_stats(mock_db, user_id)
    
    # Assert
    assert result["varieties"] == []
    assert result["maturities"] == []


# --------------------------------------------------------------------------------------------------------
def test_get_user_classifications_stats_structure(mock_db):
    """
    Test 3: Vérification de la structure des données
    """
    # Arrange
    user_id = 1
    
    mock_varieties = [Mock(variety="Test", count=5, avg_score=0.9)]
    mock_maturities = [Mock(maturity="Stage1", count=3, avg_score=0.85)]
    
    mock_variety_query = MagicMock()
    mock_variety_query.join.return_value.filter.return_value.group_by.return_value.all.return_value = mock_varieties
    mock_maturity_query = MagicMock()
    mock_maturity_query.join.return_value.filter.return_value.group_by.return_value.all.return_value = mock_maturities
    mock_db.query.return_value = mock_variety_query
    
    # Act
    result = get_user_classifications_stats(mock_db, user_id)
    
    # Assert
    assert isinstance(result["varieties"][0]["variety"], str)
    assert isinstance(result["varieties"][0]["count"], int)
    assert isinstance(result["varieties"][0]["avg_score"], float)


# =========================================================
# TESTS POUR GET_USER_WEEKLY_ACTIVITY
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_weekly_activity_success(mock_db):
    """
    Test 1: Récupération réussie de l'activité hebdomadaire
    """
    # Arrange
    user_id = 1
    
    mock_activity = [
        Mock(date=datetime(2024, 1, 10).date(), count=3),
        Mock(date=datetime(2024, 1, 12).date(), count=5)
    ]
    
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = mock_activity
    mock_db.query.return_value = mock_query
    
    with patch('app.repositories.history_repository.datetime') as mock_datetime:
        fixed_now = datetime(2024, 1, 15)
        mock_datetime.utcnow.return_value = fixed_now
        mock_datetime.timedelta = timedelta
        
        # Act
        result = get_user_weekly_activity(mock_db, user_id)
        
        # Assert
        assert len(result) == 7
        assert isinstance(result[0]["date"], str)
        assert isinstance(result[0]["count"], int)


# --------------------------------------------------------------------------------------------------------
def test_get_user_weekly_activity_no_data(mock_db):
    """
    Test 2: Aucune activité dans les 7 derniers jours
    """
    # Arrange
    user_id = 1
    
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = []
    mock_db.query.return_value = mock_query
    
    with patch('app.repositories.history_repository.datetime') as mock_datetime:
        fixed_now = datetime(2024, 1, 15)
        mock_datetime.utcnow.return_value = fixed_now
        mock_datetime.timedelta = timedelta
        
        # Act
        result = get_user_weekly_activity(mock_db, user_id)
        
        # Assert
        assert len(result) == 7
        assert all(day["count"] == 0 for day in result)


# --------------------------------------------------------------------------------------------------------
def test_get_user_weekly_activity_all_days_populated(mock_db):
    """
    Test 3: Tous les jours ont des données
    """
    # Arrange
    user_id = 1
    
    mock_activity = []
    with patch('app.repositories.history_repository.datetime') as mock_datetime:
        fixed_now = datetime(2024, 1, 15)
        mock_datetime.utcnow.return_value = fixed_now
        mock_datetime.timedelta = timedelta
        
        # Créer une activité pour chaque jour
        for i in range(7):
            date = (fixed_now - timedelta(days=6-i)).date()
            mock_activity.append(Mock(date=date, count=i+1))
        
        mock_query = MagicMock()
        mock_query.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = mock_activity
        mock_db.query.return_value = mock_query
        
        # Act
        result = get_user_weekly_activity(mock_db, user_id)
        
        # Assert
        assert len(result) == 7
        assert result[0]["count"] == 1
        assert result[6]["count"] == 7


# =========================================================
# TESTS POUR GET_USER_DETECTION_SCORES_DISTRIBUTION
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_detection_scores_distribution_success(mock_db):
    """
    Test 1: Distribution réussie des scores
    """
    # Arrange
    user_id = 1
    
    # Mock des requêtes avec différentes conditions
    def mock_scalar_side_effect(*args, **kwargs):
        return 5  # Retourner 5 pour chaque catégorie
    
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.scalar.side_effect = [5, 3, 2, 1]
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_user_detection_scores_distribution(mock_db, user_id)
    
    # Assert
    assert "excellent" in result
    assert "good" in result
    assert "fair" in result
    assert "poor" in result
    assert isinstance(result["excellent"], int)
    assert isinstance(result["good"], int)


# --------------------------------------------------------------------------------------------------------
def test_get_user_detection_scores_distribution_all_zero(mock_db):
    """
    Test 2: Aucun score de détection
    """
    # Arrange
    user_id = 1
    
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.scalar.return_value = 0
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_user_detection_scores_distribution(mock_db, user_id)
    
    # Assert
    assert result["excellent"] == 0
    assert result["good"] == 0
    assert result["fair"] == 0
    assert result["poor"] == 0


# --------------------------------------------------------------------------------------------------------
def test_get_user_detection_scores_distribution_structure(mock_db):
    """
    Test 3: Vérification de la structure des catégories
    """
    # Arrange
    user_id = 1
    
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.scalar.return_value = 10
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_user_detection_scores_distribution(mock_db, user_id)
    
    # Assert
    expected_keys = ["excellent", "good", "fair", "poor"]
    assert all(key in result for key in expected_keys)
    assert len(result) == 4


# =========================================================
# TESTS POUR GET_USER_RECENT_PREDICTIONS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_get_user_recent_predictions_success(mock_db):
    """
    Test 1: Récupération réussie des prédictions récentes
    """
    # Arrange
    user_id = 1
    
    # Utiliser des objets réels ou des mocks avec des attributs configurés
    # car le code accède à r.id, r.avg_score, etc.
    class ResultRow:
        def __init__(self, id, image, created_at, detection_count, avg_score):
            self.id = id
            self.image = image
            self.created_at = created_at
            self.detection_count = detection_count
            self.avg_score = avg_score

    mock_predictions = [
        ResultRow(1, b"img1", datetime(2024, 1, 15), 5, 0.85),
        ResultRow(2, b"img2", datetime(2024, 1, 14), 3, 0.78)
    ]
    
    mock_query = MagicMock()
    mock_query.outerjoin.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = mock_predictions
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_user_recent_predictions(mock_db, user_id, limit=5)
    
    # Assert
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[0]["detection_count"] == 5
    assert result[0]["avg_detection_score"] == 0.85


# --------------------------------------------------------------------------------------------------------
def test_get_user_recent_predictions_empty(mock_db):
    """
    Test 2: Aucune prédiction récente
    """
    # Arrange
    user_id = 1
    
    mock_query = MagicMock()
    mock_query.outerjoin.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_user_recent_predictions(mock_db, user_id)
    
    # Assert
    assert result == []


# --------------------------------------------------------------------------------------------------------
def test_get_user_recent_predictions_limit(mock_db):
    """
    Test 3: Vérification que la limite est respectée
    """
    # Arrange
    user_id = 1
    
    class ResultRow:
        def __init__(self, id, image, created_at, detection_count, avg_score):
            self.id = id
            self.image = image
            self.created_at = created_at
            self.detection_count = detection_count
            self.avg_score = avg_score

    mock_predictions = [ResultRow(i, b"img", datetime.utcnow(), 2, 0.8) for i in range(3)]
    mock_query = MagicMock()
    mock_query.outerjoin.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = mock_predictions
    mock_db.query.return_value = mock_query
    
    # Act
    result = get_user_recent_predictions(mock_db, user_id, limit=3)
    
    # Assert
    assert len(result) == 3
    # Vérifier que limit a été appelé avec 3
    mock_query.outerjoin.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.assert_called_with(3)