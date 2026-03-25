"""
Tests unitaires pour le service de détection YOLO
"""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import torch
from app.services.detection_service import detect_objects


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def real_image():
    """Crée une vraie image PIL pour les tests"""
    return Image.new('RGB', (640, 480), color='blue')


# =========================================================
# TESTS POUR DETECT_OBJECTS
# =========================================================

# --------------------------------------------------------------------------------------------------------
def test_detect_objects_success(real_image):
    """
    Test 1: Détection réussie d'objets dans une image
    """
    mock_model = Mock()
    
    mock_results = Mock()
    
    mock_box1 = Mock()
    mock_box1.xyxy = [torch.tensor([100.0, 150.0, 200.0, 250.0])]
    mock_box1.conf = [torch.tensor(0.95)]
    mock_box1.cls = [torch.tensor(0)]
    
    mock_box2 = Mock()
    mock_box2.xyxy = [torch.tensor([300.0, 350.0, 400.0, 450.0])]
    mock_box2.conf = [torch.tensor(0.87)]
    mock_box2.cls = [torch.tensor(1)]
    
    mock_boxes = Mock()
    mock_boxes.__iter__ = Mock(return_value=iter([mock_box1, mock_box2]))
    mock_results.boxes = mock_boxes
    
    mock_model.return_value = [mock_results]
    
    with patch('app.services.detection_service.get_yolo_model', return_value=mock_model):
        with patch('app.services.detection_service.DEVICE', 'cpu'):
            
            results = detect_objects(real_image)
            
            assert len(results) == 2

            assert results[0]["bbox"] == [100.0, 150.0, 200.0, 250.0]
            assert round(results[0]["score"], 2) == 0.95
            assert results[0]["class_id"] == 0
            
            assert results[1]["bbox"] == [300.0, 350.0, 400.0, 450.0]
            assert round(results[1]["score"], 2) == 0.87
            assert results[1]["class_id"] == 1
            
            mock_model.assert_called_once_with(real_image, device='cpu')


# --------------------------------------------------------------------------------------------------------
def test_detect_objects_no_detections(real_image):
    """
    Test 2: Aucune détection trouvée dans l'image
    """
    mock_model = Mock()
    

    mock_results = Mock()
    mock_results.boxes = None
    mock_model.return_value = [mock_results]
    
    with patch('app.services.detection_service.get_yolo_model', return_value=mock_model):
        with patch('app.services.detection_service.DEVICE', 'cpu'):
            results = detect_objects(real_image)
            
            assert results == []
            mock_model.assert_called_once_with(real_image, device='cpu')


# --------------------------------------------------------------------------------------------------------
def test_detect_objects_with_cuda_device(real_image):
    """
    Test 3: Détection avec device CUDA (mocké)
    """
    mock_model = Mock()
    
    mock_results = Mock()
    mock_box = Mock()
    mock_box.xyxy = [torch.tensor([50.0, 60.0, 150.0, 200.0])]
    mock_box.conf = [torch.tensor(0.75)]
    mock_box.cls = [torch.tensor(2)]
    
    mock_boxes = Mock()
    mock_boxes.__iter__ = Mock(return_value=iter([mock_box]))
    mock_results.boxes = mock_boxes
    mock_model.return_value = [mock_results]
    
    with patch('app.services.detection_service.get_yolo_model', return_value=mock_model):
        with patch('app.services.detection_service.DEVICE', 'cuda'):
            results = detect_objects(real_image)
            
            assert len(results) == 1
            assert results[0]["bbox"] == [50.0, 60.0, 150.0, 200.0]
            assert round(results[0]["score"], 2) == 0.75
            assert results[0]["class_id"] == 2
            
            mock_model.assert_called_once_with(real_image, device='cuda')
            
            
# --------------------------------------------------------------------------------------------------------