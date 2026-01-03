"""
FPSS (Fully Paired Speckle Separation) framework.

A deep learning framework for OCT image denoising and speckle separation
using attention-enhanced U-Net architectures.
"""

__version__ = "0.5.0"
__author__ = "Calvin Leighton"

# Core modules
from .models.model_factory import ModelFactory, create_model_from_config, load_model_from_checkpoint
from .losses.unified_losses import get_loss_function, FPSSLoss, MSELoss, DiceLoss, DiceBCELoss
from .data.unified_dataloader import DataLoaderFactory, SDOCTDataset, PairedOCTDataset
from .training.unified_trainer import UnifiedTrainer, TrainingConfig, load_config_from_yaml
from .utils.metrics import MetricsCalculator
from .utils.visualization import TrainingVisualizer

# Model components
from .models.components.shared_components import (
    ChannelAttention, SpatialAttention, DoubleConv, Down, Up, OutConv,
    DilatedConvBlock, ResidualBlock
)

__all__ = [
    # Core functionality
    'ModelFactory',
    'create_model_from_config',
    'load_model_from_checkpoint',
    'get_loss_function',
    'DataLoaderFactory',
    'UnifiedTrainer',
    'TrainingConfig',
    'load_config_from_yaml',
    'MetricsCalculator',
    'TrainingVisualizer',
    
    # Loss functions
    'FPSSLoss',
    'MSELoss',
    'DiceLoss',
    'DiceBCELoss',
    
    # Datasets
    'SDOCTDataset',
    'PairedOCTDataset',
    
    # Components
    'ChannelAttention',
    'SpatialAttention',
    'DoubleConv',
    'Down',
    'Up',
    'OutConv',
    'DilatedConvBlock',
    'ResidualBlock',
]