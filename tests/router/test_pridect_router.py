"""
Tests unitaires pour les routes de prédiction (version simplifiée)
"""

import pytest
from fastapi import HTTPException, UploadFile
from unittest.mock import Mock, patch, AsyncMock
import io
import numpy as np

from app.routers.pridect_router import pridect_router


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_db():
    """Mock de la session database"""
    return Mock()


@pytest.fixture
def mock_current_user():
    """Mock de l'utilisateur connecté"""
    user = Mock()
    user.id = 1
    return user


@pytest.fixture
def mock_image_file():
    """Mock d'un fichier image valide"""
    file = Mock(spec=UploadFile)
    file.content_type = "image/jpeg"
    
    async def mock_read():
        return b"fake_image_bytes"
    
    file.read = mock_read
    return file


# =========================================================
# TESTS POUR PREDICT
# =========================================================

@pytest.mark.anyio
async def test_predict_success(mock_db, mock_current_user, mock_image_file):
    """
    Test 1: Prédiction réussie avec image valide
    """
    # Arrange
    mock_prediction_result = {
        "detections": [{"bbox": [100, 150, 200, 250], "detection_score": 0.95}],
        "report": "Rapport de test",
        "annotated_image": np.zeros((100, 100, 3), dtype=np.uint8)
    }
    
    with patch('app.routers.pridect_router.run_prediction_pipeline') as mock_pipeline:
        mock_pipeline.return_value = mock_prediction_result
        
        with patch('app.routers.pridect_router.image_to_base64') as mock_to_base64:
            mock_to_base64.return_value = "base64_image"
            
            with patch('app.routers.pridect_router.save_full_result') as mock_save:
                # Act
                result = await pridect_router.routes[0].endpoint(
                    db=mock_db,
                    file=mock_image_file,
                    current_user=mock_current_user
                )
                
                # Assert
                assert result["success"] is True
                assert result["detections"] == mock_prediction_result["detections"]
                assert result["report"] == mock_prediction_result["report"]
                assert result["image"] == "base64_image"
                mock_pipeline.assert_called_once()
                mock_to_base64.assert_called_once()
                mock_save.assert_called_once()



@pytest.mark.anyio
async def test_predict_invalid_file_type(mock_db, mock_current_user):
    """
    Test 2: Rejet des fichiers non-image
    """
    # Arrange
    invalid_file = Mock(spec=UploadFile)
    invalid_file.content_type = "application/pdf"
    
    # Act & Assert
    with pytest.raises(HTTPException) as exc:
        await pridect_router.routes[0].endpoint(
            db=mock_db,
            file=invalid_file,
            current_user=mock_current_user
        )
    
    assert exc.value.status_code == 400
    assert "File must be an image" in str(exc.value.detail)



@pytest.mark.anyio
async def test_predict_pipeline_error(mock_db, mock_current_user, mock_image_file):
    """
    Test 3: Gestion d'erreur du pipeline de prédiction
    """
    # Arrange
    with patch('app.routers.pridect_router.run_prediction_pipeline') as mock_pipeline:
        mock_pipeline.side_effect = Exception("Pipeline execution failed")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc:
            await pridect_router.routes[0].endpoint(
                db=mock_db,
                file=mock_image_file,
                current_user=mock_current_user
            )
        
        assert exc.value.status_code == 500
        assert "Pipeline execution failed" in str(exc.value.detail)