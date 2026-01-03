"""
Unified loss functions for FPSS training.

This module consolidates all loss functions used in the FPSS framework
to ensure consistency and reduce code duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def __init__(self):
        """Initialize MSE loss."""
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.
        
        Args:
            input: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            MSE loss value
        """
        return self.loss_fn(input, target)


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-5):
        """Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            input: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            Dice loss value
        """
        input = torch.sigmoid(input)
        
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        intersection = (input_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross-Entropy loss."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-5):
        """Initialize combined Dice-BCE loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Smoothing factor for Dice loss
        """
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice-BCE loss.
        
        Args:
            input: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(input, target)
        bce = self.bce_loss(input, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class ContentLoss(nn.Module):
    """Content loss based on feature similarity."""
    
    def __init__(self, feature_extractor: Optional[nn.Module] = None):
        """Initialize content loss.
        
        Args:
            feature_extractor: Network to extract features (defaults to simple conv layers)
        """
        super(ContentLoss, self).__init__()
        if feature_extractor is None:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_extractor = feature_extractor
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute content loss.
        
        Args:
            input: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            Content loss value
        """
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        
        return self.mse_loss(input_features, target_features)


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel operators."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize edge loss.
        
        Args:
            weight: Weight for edge loss component
        """
        super(EdgeLoss, self).__init__()
        self.weight = weight
        
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge loss.
        
        Args:
            input: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            Edge loss value
        """
        # Compute edges
        input_edges_x = F.conv2d(input, self.sobel_x, padding=1)
        input_edges_y = F.conv2d(input, self.sobel_y, padding=1)
        input_edges = torch.sqrt(input_edges_x**2 + input_edges_y**2)
        
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        return self.weight * F.mse_loss(input_edges, target_edges)


class FPSSLoss(nn.Module):
    """Custom FPSS loss combining multiple components."""
    
    def __init__(self, 
                 alpha: float = 1.0,  # Foreground loss weight
                 beta: float = 1.0,   # Background loss weight
                 gamma: float = 1.0,  # Edge loss weight
                 delta: float = 1.0,  # Discontinuity penalty weight
                 use_content_loss: bool = True,
                 use_edge_loss: bool = True):
        """Initialize FPSS loss.
        
        Args:
            alpha: Weight for foreground loss component
            beta: Weight for background loss component
            gamma: Weight for edge loss component
            delta: Weight for discontinuity penalty
            use_content_loss: Whether to include content loss
            use_edge_loss: Whether to include edge loss
        """
        super(FPSSLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_content_loss = use_content_loss
        self.use_edge_loss = use_edge_loss
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        if use_content_loss:
            self.content_loss = ContentLoss()
        if use_edge_loss:
            self.edge_loss = EdgeLoss()
    
    def forward(self, 
                flow_pred: torch.Tensor, 
                noise_pred: torch.Tensor,
                flow_target: torch.Tensor, 
                noise_target: torch.Tensor,
                original_input: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute FPSS loss.
        
        Args:
            flow_pred: Predicted flow component
            noise_pred: Predicted noise component
            flow_target: Ground truth flow component
            noise_target: Ground truth noise component
            original_input: Original input image
            
        Returns:
            Total loss value and dictionary of individual loss components
        """
        # Foreground (flow) loss
        foreground_loss = self.mse_loss(flow_pred, flow_target)
        
        # Background (noise) loss
        background_loss = self.mse_loss(noise_pred, noise_target)
        
        # Reconstruction loss
        reconstruction = flow_pred + noise_pred
        reconstruction_loss = self.mse_loss(reconstruction, original_input)
        
        # Edge loss for flow component
        edge_loss = 0
        if self.use_edge_loss:
            edge_loss = self.edge_loss(flow_pred, flow_target)
        
        # Content loss for flow component
        content_loss = 0
        if self.use_content_loss:
            content_loss = self.content_loss(flow_pred, flow_target)
        
        # Discontinuity penalty - encourage smooth noise
        discontinuity_loss = self._compute_discontinuity_penalty(noise_pred)
        
        # Total loss
        total_loss = (self.alpha * foreground_loss +
                     self.beta * background_loss +
                     reconstruction_loss +
                     self.gamma * edge_loss +
                     content_loss +
                     self.delta * discontinuity_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'foreground_loss': foreground_loss,
            'background_loss': background_loss,
            'reconstruction_loss': reconstruction_loss,
            'edge_loss': edge_loss,
            'content_loss': content_loss,
            'discontinuity_loss': discontinuity_loss
        }
        
        return total_loss, loss_dict
    
    def _compute_discontinuity_penalty(self, noise: torch.Tensor) -> torch.Tensor:
        """Compute discontinuity penalty for noise component.
        
        Args:
            noise: Predicted noise tensor
            
        Returns:
            Discontinuity penalty value
        """
        # Gradient in x direction
        grad_x = torch.abs(noise[:, :, :, 1:] - noise[:, :, :, :-1])
        
        # Gradient in y direction
        grad_y = torch.abs(noise[:, :, 1:, :] - noise[:, :, :-1, :])
        
        # Total variation penalty
        tv_loss = torch.mean(grad_x) + torch.mean(grad_y)
        
        return tv_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features."""
    
    def __init__(self, layers: list = ['relu1_1', 'relu2_1', 'relu3_1'], 
                 weights: list = [1.0, 1.0, 1.0]):
        """Initialize perceptual loss.
        
        Args:
            layers: VGG layers to use for feature extraction
            weights: Weights for each layer
        """
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.weights = weights
        
        # Note: This is a simplified version - actual implementation would use VGG features
        # For OCT images, we might want to use a different feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            input: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            Perceptual loss value
        """
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        
        return self.mse_loss(input_features, target_features)


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss function name is not recognized
    """
    loss_functions = {
        'mse': MSELoss,
        'dice': DiceLoss,
        'dice_bce': DiceBCELoss,
        'content': ContentLoss,
        'edge': EdgeLoss,
        'fpss': FPSSLoss,
        'perceptual': PerceptualLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name](**kwargs)