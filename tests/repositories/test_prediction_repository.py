"""
Tests unitaires pour le pipeline de prédiction
"""

import pytest
import cv2
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from app.repositories.prediction_repository import run_prediction_pipeline, VARIETY_LABELS, MATURITY_LABELS


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_image_file():
    """Mock d'un fichier image"""
    file = Mock()
    file.read.return_value = b"fake_image_data"
    return file


@pytest.fixture
def sample_detections():
    """Exemple de détections YOLO"""
    return [
        {
            "bbox": [100, 150, 200, 250],
            "score": 0.95,
            "class_id": 0
        },
        {
            "bbox": [300, 350, 400, 450],
            "score": 0.87,
            "class_id": 1
        }
    ]


@pytest.fixture
def sample_variety_result():
    """Exemple de résultat de classification variété"""
    return {
        "class_id": 0,
        "class_name": "Boufagous",
        "confidence": 0.92
    }


@pytest.fixture
def sample_maturity_result():
    """Exemple de résultat de classification maturité"""
    return {
        "class_id": 2,
        "class_name": "Stage 3",
        "confidence": 0.88
    }


# =========================================================
# TESTS POUR RUN_PREDICTION_PIPELINE
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_run_prediction_pipeline_success(mock_image_file, sample_detections, sample_variety_result, sample_maturity_result):
    """
    Test 1: Pipeline complet réussi avec détections
    """
    # Arrange - Créer une vraie image PIL
    pil_image = Image.new('RGB', (640, 480), color='red')
    
    with patch('app.repositories.prediction_repository.load_image', return_value=pil_image):
        with patch('app.repositories.prediction_repository.pil_to_numpy') as mock_pil_to_numpy:
            mock_pil_to_numpy.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Mock de detect_objects
            with patch('app.repositories.prediction_repository.detect_objects', return_value=sample_detections):
                # Mock de predict_variety
                with patch('app.repositories.prediction_repository.predict_variety', return_value=sample_variety_result):
                    # Mock de predict_maturity
                    with patch('app.repositories.prediction_repository.predict_maturity', return_value=sample_maturity_result):
                        # Mock de generate_report
                        with patch('app.repositories.prediction_repository.generate_report', return_value="Rapport de test"):
                            
                            # Act
                            result = run_prediction_pipeline(mock_image_file)
                            
                            # Assert
                            assert "detections" in result
                            assert "report" in result
                            assert "annotated_image" in result
                            assert len(result["detections"]) == 2
                            assert result["report"] == "Rapport de test"
                            assert isinstance(result["annotated_image"], np.ndarray)


# --------------------------------------------------------------------------------------------------------
def test_run_prediction_pipeline_no_detections(mock_image_file):
    """
    Test 2: Pipeline avec aucune détection
    """
    # Arrange
    pil_image = Image.new('RGB', (640, 480), color='red')
    
    with patch('app.repositories.prediction_repository.load_image', return_value=pil_image):
        with patch('app.repositories.prediction_repository.pil_to_numpy', return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            with patch('app.repositories.prediction_repository.detect_objects', return_value=[]):
                with patch('app.repositories.prediction_repository.generate_report', return_value="Aucune détection"):
                    
                    # Act
                    result = run_prediction_pipeline(mock_image_file)
                    
                    # Assert
                    assert result["detections"] == []
                    assert result["report"] == "Aucune détection"
                    assert isinstance(result["annotated_image"], np.ndarray)


# --------------------------------------------------------------------------------------------------------
def test_run_prediction_pipeline_invalid_bbox(mock_image_file, sample_detections, sample_variety_result, sample_maturity_result):
    """
    Test 3: Gestion des bounding boxes invalides (coordonnées négatives ou inversées)
    """
    # Arrange
    pil_image = Image.new('RGB', (640, 480), color='red')
    
    # Détections avec bbox invalide
    invalid_detections = [
        {
            "bbox": [-20, -20, -10, -10],  # Entièrement hors image (négatif)
            "score": 0.95,
            "class_id": 0
        },
        {
            "bbox": [100, 150, 50, 250],  # x2 < x1
            "score": 0.87,
            "class_id": 1
        },
        {
            "bbox": [300, 350, 400, 450],  # Bbox valide
            "score": 0.92,
            "class_id": 2
        }
    ]
    
    with patch('app.repositories.prediction_repository.load_image', return_value=pil_image):
        with patch('app.repositories.prediction_repository.pil_to_numpy', return_value=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)):
            with patch('app.repositories.prediction_repository.detect_objects', return_value=invalid_detections):
                with patch('app.repositories.prediction_repository.predict_variety', return_value=sample_variety_result):
                    with patch('app.repositories.prediction_repository.predict_maturity', return_value=sample_maturity_result):
                        with patch('app.repositories.prediction_repository.generate_report', return_value="Rapport avec bbox valides"):
                            
                            # Act
                            result = run_prediction_pipeline(mock_image_file)
                            
                            # Assert
                            # Seule la bbox valide doit être traitée
                            assert len(result["detections"]) == 1
                            assert result["detections"][0]["bbox"] == [300, 350, 400, 450]
                            assert result["detections"][0]["variety"] == "Boufagous"
                            assert result["detections"][0]["maturity"] == "Stage 3"