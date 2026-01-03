"""
Metrics calculation utilities for FPSS evaluation.

This module provides comprehensive metrics for evaluating OCT image denoising
and speckle separation performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, peak_signal_noise_ratio
import cv2


class MetricsCalculator:
    """Comprehensive metrics calculator for FPSS evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = []
    
    def calculate_all_metrics(self, 
                            prediction: torch.Tensor, 
                            target: torch.Tensor,
                            input_image: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Calculate all available metrics.
        
        Args:
            prediction: Predicted image tensor
            target: Ground truth image tensor
            input_image: Original input image (optional)
            
        Returns:
            Dictionary of metric values
        """
        # Convert to numpy for some metrics
        pred_np = self._tensor_to_numpy(prediction)
        target_np = self._tensor_to_numpy(target)
        
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = self.calculate_mse(prediction, target)
        metrics['mae'] = self.calculate_mae(prediction, target)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Image quality metrics
        metrics['psnr'] = self.calculate_psnr(prediction, target)
        metrics['ssim'] = self.calculate_ssim(prediction, target)
        
        # Structural metrics
        metrics['structural_similarity'] = self.calculate_structural_similarity(pred_np, target_np)
        
        # Edge metrics
        metrics['edge_preservation'] = self.calculate_edge_preservation(pred_np, target_np)
        
        # Contrast metrics
        metrics['contrast_preservation'] = self.calculate_contrast_preservation(pred_np, target_np)
        
        # Noise metrics
        if input_image is not None:
            input_np = self._tensor_to_numpy(input_image)
            metrics['noise_reduction'] = self.calculate_noise_reduction(input_np, pred_np, target_np)
            metrics['snr_improvement'] = self.calculate_snr_improvement(input_np, pred_np)
        
        # Vessel-specific metrics (for OCT/OCTA)
        metrics['vessel_continuity'] = self.calculate_vessel_continuity(pred_np, target_np)
        metrics['vessel_contrast'] = self.calculate_vessel_contrast(pred_np, target_np)
        
        return metrics
    
    def calculate_mse(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Mean Squared Error.
        
        Args:
            prediction: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            MSE value
        """
        return F.mse_loss(prediction, target).item()
    
    def calculate_mae(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Mean Absolute Error.
        
        Args:
            prediction: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            MAE value
        """
        return F.l1_loss(prediction, target).item()
    
    def calculate_psnr(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            prediction: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(prediction, target)
        if mse == 0:
            return float('inf')
        
        max_pixel = 1.0  # Assuming normalized images
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    def calculate_ssim(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Structural Similarity Index.
        
        Args:
            prediction: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            SSIM value
        """
        # Convert to numpy for SSIM calculation
        pred_np = self._tensor_to_numpy(prediction)
        target_np = self._tensor_to_numpy(target)
        
        # Calculate SSIM using scikit-image
        try:
            from skimage.metrics import structural_similarity
            ssim_val = structural_similarity(target_np, pred_np, data_range=1.0)
            return ssim_val
        except ImportError:
            # Fallback to PyTorch implementation
            return self._pytorch_ssim(prediction, target)
    
    def _pytorch_ssim(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """PyTorch implementation of SSIM.
        
        Args:
            prediction: Predicted tensor
            target: Ground truth tensor
            
        Returns:
            SSIM value
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(prediction, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(prediction * prediction, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(prediction * target, 3, 1, 1) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def calculate_structural_similarity(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate structural similarity between images.
        
        Args:
            prediction: Predicted image array
            target: Ground truth image array
            
        Returns:
            Structural similarity value
        """
        # Normalize images
        pred_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        target_norm = (target - target.min()) / (target.max() - target.min())
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(pred_norm.flatten(), target_norm.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_edge_preservation(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate edge preservation metric.
        
        Args:
            prediction: Predicted image array
            target: Ground truth image array
            
        Returns:
            Edge preservation value
        """
        # Calculate edges using Sobel operator
        pred_edges = self._calculate_edges(prediction)
        target_edges = self._calculate_edges(target)
        
        # Calculate correlation between edge maps
        correlation = np.corrcoef(pred_edges.flatten(), target_edges.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_edges(self, image: np.ndarray) -> np.ndarray:
        """Calculate edge magnitude using Sobel operator.
        
        Args:
            image: Input image array
            
        Returns:
            Edge magnitude array
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Calculate gradients
        grad_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return magnitude
    
    def calculate_contrast_preservation(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate contrast preservation metric.
        
        Args:
            prediction: Predicted image array
            target: Ground truth image array
            
        Returns:
            Contrast preservation value
        """
        # Calculate local contrast
        pred_contrast = self._calculate_local_contrast(prediction)
        target_contrast = self._calculate_local_contrast(target)
        
        # Calculate correlation
        correlation = np.corrcoef(pred_contrast.flatten(), target_contrast.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_local_contrast(self, image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate local contrast using sliding window.
        
        Args:
            image: Input image array
            window_size: Size of the sliding window
            
        Returns:
            Local contrast array
        """
        # Pad image
        pad_size = window_size // 2
        padded = np.pad(image, pad_size, mode='reflect')
        
        # Calculate local contrast
        contrast = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                contrast[i, j] = np.std(window)
        
        return contrast
    
    def calculate_noise_reduction(self, input_image: np.ndarray, 
                                prediction: np.ndarray, 
                                target: np.ndarray) -> float:
        """Calculate noise reduction metric.
        
        Args:
            input_image: Original noisy image
            prediction: Denoised prediction
            target: Ground truth clean image
            
        Returns:
            Noise reduction value
        """
        # Calculate noise in input and prediction
        input_noise = np.std(input_image - target)
        pred_noise = np.std(prediction - target)
        
        # Calculate noise reduction ratio
        if input_noise == 0:
            return 0.0
        
        noise_reduction = (input_noise - pred_noise) / input_noise
        
        return noise_reduction
    
    def calculate_snr_improvement(self, input_image: np.ndarray, prediction: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio improvement.
        
        Args:
            input_image: Original noisy image
            prediction: Denoised prediction
            
        Returns:
            SNR improvement in dB
        """
        # Calculate signal power (mean squared intensity)
        signal_power = np.mean(prediction**2)
        
        # Calculate noise power (variance of the difference)
        noise_power = np.var(input_image - prediction)
        
        if noise_power == 0:
            return float('inf')
        
        # Calculate SNR improvement
        snr_improvement = 10 * np.log10(signal_power / noise_power)
        
        return snr_improvement
    
    def calculate_vessel_continuity(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate vessel continuity metric (specific to OCT/OCTA).
        
        Args:
            prediction: Predicted image array
            target: Ground truth image array
            
        Returns:
            Vessel continuity value
        """
        # Threshold images to create binary vessel maps
        pred_binary = prediction > np.percentile(prediction, 90)
        target_binary = target > np.percentile(target, 90)
        
        # Calculate connected components
        pred_components = self._get_connected_components(pred_binary)
        target_components = self._get_connected_components(target_binary)
        
        # Calculate overlap between components
        overlap = np.sum(pred_binary & target_binary)
        total = np.sum(pred_binary | target_binary)
        
        if total == 0:
            return 0.0
        
        return overlap / total
    
    def _get_connected_components(self, binary_image: np.ndarray) -> int:
        """Get number of connected components in binary image.
        
        Args:
            binary_image: Binary image array
            
        Returns:
            Number of connected components
        """
        # Convert to uint8
        binary_uint8 = binary_image.astype(np.uint8)
        
        # Find connected components
        num_components, _ = cv2.connectedComponents(binary_uint8)
        
        return num_components
    
    def calculate_vessel_contrast(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate vessel contrast metric.
        
        Args:
            prediction: Predicted image array
            target: Ground truth image array
            
        Returns:
            Vessel contrast value
        """
        # Calculate vessel regions (high intensity areas)
        pred_vessels = prediction > np.percentile(prediction, 85)
        target_vessels = target > np.percentile(target, 85)
        
        # Calculate background regions
        pred_background = prediction < np.percentile(prediction, 15)
        target_background = target < np.percentile(target, 15)
        
        # Calculate contrast ratios
        pred_contrast = np.mean(prediction[pred_vessels]) - np.mean(prediction[pred_background])
        target_contrast = np.mean(target[target_vessels]) - np.mean(target[target_background])
        
        # Calculate contrast preservation
        if target_contrast == 0:
            return 0.0
        
        contrast_ratio = pred_contrast / target_contrast
        
        return min(contrast_ratio, 1.0)  # Cap at 1.0
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        # Move to CPU if on GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert to numpy
        array = tensor.numpy()
        
        # Remove batch and channel dimensions if present
        if array.ndim == 4:  # Batch, Channel, Height, Width
            array = array[0, 0]
        elif array.ndim == 3:  # Channel, Height, Width or Batch, Height, Width
            array = array[0] if array.shape[0] == 1 else array
        
        return array
    
    def update_history(self, metrics: Dict[str, float]):
        """Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
        """
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics over all history.
        
        Returns:
            Dictionary of average metric values
        """
        if not self.metrics_history:
            return {}
        
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def get_metrics_summary(self) -> str:
        """Get summary string of metrics.
        
        Returns:
            Summary string
        """
        if not self.metrics_history:
            return "No metrics available"
        
        avg_metrics = self.get_average_metrics()
        
        summary = "Metrics Summary:\n"
        summary += "-" * 40 + "\n"
        
        for key, value in avg_metrics.items():
            summary += f"{key:20}: {value:.4f}\n"
        
        return summary