"""
Tests unitaires pour le service de chargement d'images
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
import cv2

from app.services.upload_img_service import (
    load_image,
    pil_to_numpy,
    rgb_to_bgr,
    preprocess_for_detection,
    get_classification_transform,
    preprocess_for_classification
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_pil_image():
    """Crée une image PIL mockée"""
    image = Mock(spec=Image.Image)
    image.convert.return_value = image
    return image


@pytest.fixture
def real_pil_image():
    """Crée une vraie image PIL pour les tests d'intégration"""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def mock_image_file():
    """Mock d'un fichier image"""
    file = Mock()
    file.read.return_value = b"fake_image_data"
    return file


@pytest.fixture
def sample_numpy_array():
    """Crée un tableau numpy de test"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


# =========================================================
# TESTS LOAD_IMAGE
# =========================================================
def test_load_image_from_pil(mock_pil_image):
    """Test chargement depuis une image PIL"""
    result = load_image(mock_pil_image)
    
    assert result == mock_pil_image
    mock_pil_image.convert.assert_called_once_with("RGB")


def test_load_image_from_file(mock_image_file):
    """Test chargement depuis un fichier"""
    with patch('PIL.Image.open') as mock_open:
        mock_pil = Mock(spec=Image.Image)
        mock_pil.convert.return_value = mock_pil
        mock_open.return_value = mock_pil
        
        result = load_image(mock_image_file)
        
        mock_open.assert_called_once_with(mock_image_file)
        mock_pil.convert.assert_called_once_with("RGB")
        assert result == mock_pil


def test_load_image_converts_to_rgb(real_pil_image):
    """Test que l'image est bien convertie en RGB"""
    img_mode_p = real_pil_image.convert('P')
    result = load_image(img_mode_p)
    
    assert result.mode == 'RGB'


# =========================================================
# TESTS CONVERSIONS
# =========================================================

def test_pil_to_numpy(real_pil_image):
    """Test conversion PIL vers numpy"""
    result = pil_to_numpy(real_pil_image)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_pil_to_numpy_empty():
    """Test avec une image vide"""
    empty_image = Image.new('RGB', (0, 0))
    result = pil_to_numpy(empty_image)
    
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_rgb_to_bgr(sample_numpy_array):
    """Test conversion RGB vers BGR"""
    result = rgb_to_bgr(sample_numpy_array)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_numpy_array.shape
    np.testing.assert_array_equal(result[:, :, 0], sample_numpy_array[:, :, 2])
    np.testing.assert_array_equal(result[:, :, 2], sample_numpy_array[:, :, 0])


def test_rgb_to_bgr_with_cv2_mock(sample_numpy_array):
    """Test conversion avec cv2 mocké"""
    with patch('cv2.cvtColor') as mock_cvtColor:
        mock_cvtColor.return_value = sample_numpy_array
        result = rgb_to_bgr(sample_numpy_array)
        
        mock_cvtColor.assert_called_once_with(
            sample_numpy_array,
            cv2.COLOR_RGB2BGR
        )
        assert np.array_equal(result, sample_numpy_array)


# =========================================================
# TESTS PREPROCESS FOR DETECTION
# =========================================================

def test_preprocess_for_detection_success(mock_image_file):
    """Test preprocess pour détection réussi"""
    mock_pil = Mock(spec=Image.Image)
    mock_numpy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    with patch('app.services.upload_img_service.load_image') as mock_load:
        with patch('app.services.upload_img_service.pil_to_numpy') as mock_pil_to_numpy:
            with patch('app.services.upload_img_service.rgb_to_bgr') as mock_rgb_to_bgr:
                mock_load.return_value = mock_pil
                mock_pil_to_numpy.return_value = mock_numpy
                mock_rgb_to_bgr.return_value = mock_numpy
                
                result = preprocess_for_detection(mock_image_file)
                
                mock_load.assert_called_once_with(mock_image_file)
                mock_pil_to_numpy.assert_called_once_with(mock_pil)
                mock_rgb_to_bgr.assert_called_once_with(mock_numpy)
                assert isinstance(result, np.ndarray)


def test_preprocess_for_detection_integration(real_pil_image):
    """Test d'intégration du preprocess pour détection"""
    result = preprocess_for_detection(real_pil_image)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


# =========================================================
# TESTS CLASSIFICATION TRANSFORM
# =========================================================

def test_get_classification_transform():
    """Test création du transform"""
    transform = get_classification_transform()
    
    assert transform is not None
    assert len(transform.transforms) == 3
    assert isinstance(transform.transforms[0], transforms.Resize)
    assert isinstance(transform.transforms[1], transforms.ToTensor)
    assert isinstance(transform.transforms[2], transforms.Normalize)


def test_transform_parameters():
    """Test paramètres du transform"""
    transform = get_classification_transform()
    
    resize = transform.transforms[0]
    assert resize.size == (224, 224)
    
    normalize = transform.transforms[2]
    assert normalize.mean == [0.485, 0.456, 0.406]
    assert normalize.std == [0.229, 0.224, 0.225]


# =========================================================
# TESTS PREPROCESS FOR CLASSIFICATION
# =========================================================

def test_preprocess_with_pil_image(real_pil_image):
    """Test preprocess avec image PIL"""
    device = torch.device('cpu')
    result = preprocess_for_classification(real_pil_image, device)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 224, 224)
    assert result.device.type == 'cpu'


def test_preprocess_with_file():
    """Test preprocess avec fichier"""
    image = Image.new('RGB', (100, 100), color='blue')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    device = torch.device('cpu')
    result = preprocess_for_classification(buffer, device)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 224, 224)
    assert result.device.type == 'cpu'


def test_preprocess_tensor_shape(real_pil_image):
    """Test que le tensor a la bonne forme"""
    device = torch.device('cpu')
    result = preprocess_for_classification(real_pil_image, device)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 224, 224)
    assert result.device == device


def test_preprocess_with_invalid_input():
    """Test avec input invalide"""
    device = torch.device('cpu')
    with pytest.raises(Exception):
        preprocess_for_classification(None, device)


def test_preprocess_normalization_values(real_pil_image):
    """Test que les valeurs sont normalisées"""
    device = torch.device('cpu')
    result = preprocess_for_classification(real_pil_image, device)
    
    assert result.min() < 0
    assert result.max() > 0
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


# =========================================================
# TESTS D'INTÉGRATION
# =========================================================

def test_full_pipeline_with_real_image():
    """Test complet du pipeline avec une vraie image"""
    image = Image.new('RGB', (300, 300), color='blue')
    
    detection_result = preprocess_for_detection(image)
    assert isinstance(detection_result, np.ndarray)
    assert detection_result.shape == (300, 300, 3)
    
    device = torch.device('cpu')
    classification_result = preprocess_for_classification(image, device)
    assert isinstance(classification_result, torch.Tensor)
    assert classification_result.shape == (1, 3, 224, 224)


def test_with_bytesio_input():
    """Test avec BytesIO comme input"""
    image = Image.new('RGB', (100, 100), color='green')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    result = load_image(buffer)
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGB'


@patch('torch.cuda.is_available')
def test_with_cuda_device(mock_cuda):
    """Test avec device CUDA"""
    mock_cuda.return_value = True
    image = Image.new('RGB', (100, 100), color='red')
    device = torch.device('cuda')
    
    with patch('app.services.upload_img_service.get_classification_transform') as mock_get_transform:
        mock_transform = MagicMock()
        mock_tensor = MagicMock()
        mock_get_transform.return_value = mock_transform
        mock_transform.return_value = mock_tensor
        
        with patch('app.services.upload_img_service.load_image') as mock_load:
            result = preprocess_for_classification(image, device)
            
            mock_transform.assert_called_once_with(image)
            mock_tensor.unsqueeze.assert_called_once_with(0)
            mock_tensor.unsqueeze.return_value.to.assert_called_once_with(device)
            assert result == mock_tensor.unsqueeze.return_value.to.return_value


def test_cuda_device_integration():
    """Test d'intégration avec device CUDA (skip si CUDA non disponible)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    image = Image.new('RGB', (100, 100), color='red')
    device = torch.device('cuda')
    result = preprocess_for_classification(image, device)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 224, 224)
    assert result.device.type == 'cuda'


# =========================================================
# TESTS ADDITIONNELS POUR LES CAS LIMITES
# =========================================================

def test_load_image_with_none():
    """Test load_image avec None"""
    with pytest.raises(Exception):
        load_image(None)


def test_pil_to_numpy_with_none():
    """Test pil_to_numpy avec None"""
    with pytest.raises(TypeError):
        pil_to_numpy(None)


def test_rgb_to_bgr_with_invalid_shape():
    """Test rgb_to_bgr avec une forme invalide"""
    invalid_array = np.random.randint(0, 255, (100, 100))
    with pytest.raises(Exception):
        rgb_to_bgr(invalid_array)


def test_preprocess_for_classification_with_empty_image():
    """Test preprocess avec une image vide"""
    empty_image = Image.new('RGB', (0, 0))
    device = torch.device('cpu')
    
    result = preprocess_for_classification(empty_image, device)
    assert isinstance(result, torch.Tensor)
    assert len(result.shape) >= 1


def test_multiple_transforms_consistency():
    """Test la cohérence des transformations multiples"""
    transform = get_classification_transform()
    
    transform2 = get_classification_transform()
    assert transform is not transform2
    assert transform.transforms[0].size == transform2.transforms[0].size


# =========================================================
# TESTS AVEC DES MOCKS AMÉLIORÉS
# =========================================================

def test_preprocess_with_pil_image_mocked():
    """Test preprocess avec image PIL et mocks améliorés"""
    with patch('app.services.upload_img_service.get_classification_transform') as mock_get_transform:
        mock_transform = MagicMock()
        mock_tensor = MagicMock()
        mock_transform.return_value = mock_tensor
        mock_get_transform.return_value = mock_transform
        
        device = torch.device('cpu')
        
        image = Image.new('RGB', (100, 100), color='red')
        
        result = preprocess_for_classification(image, device)
        
        mock_transform.assert_called_once_with(image)
        mock_tensor.unsqueeze.assert_called_once_with(0)
        mock_tensor.unsqueeze.return_value.to.assert_called_once_with(device)
        assert result == mock_tensor.unsqueeze.return_value.to.return_value


def test_preprocess_with_file_mocked():
    """Test preprocess avec fichier et mocks améliorés"""
    with patch('app.services.upload_img_service.load_image') as mock_load:
        with patch('app.services.upload_img_service.get_classification_transform') as mock_get_transform:
            mock_pil = Mock(spec=Image.Image)
            mock_load.return_value = mock_pil
            
            mock_transform = MagicMock()
            mock_tensor = MagicMock()
            mock_transform.return_value = mock_tensor
            mock_get_transform.return_value = mock_transform
            
            device = torch.device('cpu')
            
            mock_file = Mock()
            
            result = preprocess_for_classification(mock_file, device)
            
            mock_load.assert_called_once_with(mock_file)
            mock_transform.assert_called_once_with(mock_pil)
            mock_tensor.unsqueeze.assert_called_once_with(0)
            mock_tensor.unsqueeze.return_value.to.assert_called_once_with(device)
            assert result == mock_tensor.unsqueeze.return_value.to.return_value


def test_preprocess_with_mock_device_real_transform():
    """Test avec un device mocké mais un vrai transform"""
    with patch('app.services.upload_img_service.get_classification_transform') as mock_get_transform:
        real_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mock_get_transform.return_value = real_transform
        
        mock_device = MagicMock()
        mock_device.type = 'cuda'
        
        image = Image.new('RGB', (100, 100), color='red')
        
        with patch('torch.Tensor.to') as mock_to:
            mock_to.return_value = torch.randn(1, 3, 224, 224)
            result = preprocess_for_classification(image, mock_device)
            mock_to.assert_called()
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 224, 224)