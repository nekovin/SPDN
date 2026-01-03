"""
Shared neural network components for FPSS models.

This module provides common building blocks used across different model architectures
to ensure consistency and reduce code duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism using CBAM approach.
    
    Emphasizes important feature channels using both average and max pooling
    followed by a shared MLP with batch normalization.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16, scale_factor: float = 1.0):
        """Initialize channel attention module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for the MLP bottleneck
            scale_factor: Scaling factor for attention weights
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.scale_factor = scale_factor
        
        # Shared MLP with batch normalization for better training stability
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted input tensor
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine channel attention
        out = avg_out + max_out
        
        # Apply attention with controlled scale factor
        attention = self.activation(out * self.scale_factor)
        
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention mechanism using CBAM approach.
    
    Emphasizes important spatial regions using channel-wise statistics.
    """
    
    def __init__(self, kernel_size: int = 7):
        """Initialize spatial attention module.
        
        Args:
            kernel_size: Kernel size for spatial attention convolution
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted input tensor
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Combine and apply spatial attention
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.activation(self.conv(out))
        
        return x * attention


class DoubleConv(nn.Module):
    """Standard U-Net double convolution block.
    
    Applies two 3x3 convolutions with BatchNorm and ReLU activations.
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (defaults to out_channels)
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after double convolution
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block with max pooling and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through downsampling block.
        
        Args:
            x: Input tensor
            
        Returns:
            Downsampled tensor
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connections and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling (vs transposed convolution)
        """
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampling block.
        
        Args:
            x1: Input tensor from previous layer
            x2: Skip connection tensor from encoder
            
        Returns:
            Upsampled tensor with skip connection
        """
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer for final predictions."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize output convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output convolution.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        return self.conv(x)


class DilatedConvBlock(nn.Module):
    """Multi-scale dilated convolution block for capturing features at different scales."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: list = [1, 2, 4]):
        """Initialize dilated convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation_rates: List of dilation rates to use
        """
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(dilation_rates), 
                     kernel_size=3, padding=rate, dilation=rate, bias=False)
            for rate in dilation_rates
        ])
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dilated convolution block.
        
        Args:
            x: Input tensor
            
        Returns:
            Multi-scale feature tensor
        """
        features = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(features, dim=1)
        out = self.batch_norm(out)
        return self.activation(out)


class ResidualBlock(nn.Module):
    """Residual block with optional bottleneck and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 use_attention: bool = False):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            use_attention: Whether to apply channel attention
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = ChannelAttention(out_channels) if use_attention else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.attention is not None:
            out = self.attention(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out