"""
Model factory for FPSS framework.

This module provides a centralized way to instantiate different model architectures
with consistent interfaces and configuration management.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod

# Import all model architectures
from .fpss.fpss_attention import FPSSAttention
from .fpss.fpss_attention_small import SmallFPSSAttention
from .fpss.fpss_no_attention import FPSSNoAttention
from .unet.unet import UNet
from .unet.simple_unet import SimpleUNet


class ModelConfig:
    """Configuration class for model parameters."""
    
    def __init__(self, **kwargs):
        """Initialize model configuration.
        
        Args:
            **kwargs: Model configuration parameters
        """
        self.config = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """Update configuration.
        
        Args:
            **kwargs: Configuration updates
        """
        self.config.update(kwargs)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes.
        
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)


class FPSSModelWrapper(BaseModel):
    """Wrapper for FPSS models to provide consistent interface."""
    
    def __init__(self, model_class: Type[nn.Module], config: ModelConfig):
        """Initialize FPSS model wrapper.
        
        Args:
            model_class: FPSS model class
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract model-specific parameters
        self.model = model_class(
            n_channels=config.get('n_channels', 1),
            n_classes=config.get('n_classes', 1),
            initial_features=config.get('initial_features', 32),
            depth=config.get('depth', 4),
            block_depth=config.get('block_depth', 3),
            bilinear=config.get('bilinear', True)
        )
        
        self._model_name = model_class.__name__
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FPSS model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output (flow and noise components)
        """
        return self.model(x)
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class UNetModelWrapper(BaseModel):
    """Wrapper for U-Net models to provide consistent interface."""
    
    def __init__(self, model_class: Type[nn.Module], config: ModelConfig):
        """Initialize U-Net model wrapper.
        
        Args:
            model_class: U-Net model class
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract model-specific parameters
        self.model = model_class(
            n_channels=config.get('n_channels', 1),
            n_classes=config.get('n_classes', 1),
            bilinear=config.get('bilinear', True)
        )
        
        self._model_name = model_class.__name__
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class ModelFactory:
    """Factory class for creating model instances."""
    
    # Registry of available models
    _models = {
        # FPSS models
        'fpss_attention': (FPSSAttention, FPSSModelWrapper),
        'fpss_attention_small': (SmallFPSSAttention, FPSSModelWrapper),
        'fpss_no_attention': (FPSSNoAttention, FPSSModelWrapper),
        
        # U-Net models
        'unet': (UNet, UNetModelWrapper),
        'simple_unet': (SimpleUNet, UNetModelWrapper),
    }
    
    # Default configurations for each model
    _default_configs = {
        'fpss_attention': {
            'n_channels': 1,
            'n_classes': 1,
            'initial_features': 32,
            'depth': 4,
            'block_depth': 3,
            'bilinear': True
        },
        'fpss_attention_small': {
            'n_channels': 1,
            'n_classes': 1,
            'initial_features': 16,
            'depth': 3,
            'block_depth': 2,
            'bilinear': True
        },
        'fpss_no_attention': {
            'n_channels': 1,
            'n_classes': 1,
            'initial_features': 32,
            'depth': 5,
            'block_depth': 3,
            'bilinear': True
        },
        'unet': {
            'n_channels': 1,
            'n_classes': 1,
            'bilinear': True
        },
        'simple_unet': {
            'n_channels': 1,
            'n_classes': 1,
            'bilinear': True
        }
    }
    
    @classmethod
    def create_model(cls, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Optional configuration dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not recognized
        """
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        # Get model class and wrapper
        model_class, wrapper_class = cls._models[model_name]
        
        # Create configuration
        default_config = cls._default_configs.get(model_name, {})
        if config:
            default_config.update(config)
        
        model_config = ModelConfig(**default_config)
        
        # Create and return wrapped model
        return wrapper_class(model_class, model_config)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names.
        
        Returns:
            List of available model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[nn.Module], 
                      wrapper_class: Type[BaseModel], default_config: Dict[str, Any] = None):
        """Register a new model with the factory.
        
        Args:
            model_name: Name to register the model under
            model_class: Model class
            wrapper_class: Wrapper class for the model
            default_config: Default configuration for the model
        """
        cls._models[model_name] = (model_class, wrapper_class)
        if default_config:
            cls._default_configs[model_name] = default_config
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
            
        Raises:
            ValueError: If model name is not recognized
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class, wrapper_class = cls._models[model_name]
        default_config = cls._default_configs.get(model_name, {})
        
        return {
            'model_name': model_name,
            'model_class': model_class.__name__,
            'wrapper_class': wrapper_class.__name__,
            'default_config': default_config,
            'description': model_class.__doc__ or 'No description available'
        }


def create_model_from_config(config_dict: Dict[str, Any]) -> BaseModel:
    """Create a model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary containing model parameters
        
    Returns:
        Model instance
        
    Raises:
        KeyError: If 'model_name' is not in configuration
        ValueError: If model name is not recognized
    """
    if 'model_name' not in config_dict:
        raise KeyError("Configuration must contain 'model_name'")
    
    model_name = config_dict.pop('model_name')
    return ModelFactory.create_model(model_name, config_dict)


def load_model_from_checkpoint(checkpoint_path: str, model_name: str, 
                             config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Name of the model architecture
        config: Optional configuration dictionary
        
    Returns:
        Model instance loaded with checkpoint weights
    """
    # Create model
    model = ModelFactory.create_model(model_name, config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    return model