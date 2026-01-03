"""
Visualization utilities for FPSS training and evaluation.

This module provides comprehensive visualization tools for training progress,
model outputs, and evaluation results.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from pathlib import Path
import json


class TrainingVisualizer:
    """Visualizer for training progress and results."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize training visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, 
                           train_losses: List[float], 
                           val_losses: List[float] = None,
                           learning_rates: List[float] = None,
                           save_path: str = None) -> None:
        """Plot training curves.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses (optional)
            learning_rates: List of learning rates (optional)
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2 if learning_rates else 1, figsize=(15, 5))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot loss curves
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses and len(val_losses) > 0:
            val_epochs = np.linspace(1, len(train_losses), len(val_losses))
            axes[0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot learning rate if provided
        if learning_rates and len(axes) > 1:
            axes[1].plot(epochs, learning_rates, 'g-', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_model_predictions(self, 
                             input_images: torch.Tensor,
                             predictions: torch.Tensor,
                             targets: torch.Tensor = None,
                             num_samples: int = 4,
                             save_path: str = None) -> None:
        """Plot model predictions.
        
        Args:
            input_images: Input images tensor
            predictions: Model predictions tensor
            targets: Ground truth targets (optional)
            num_samples: Number of samples to visualize
            save_path: Path to save the plot
        """
        # Convert tensors to numpy
        input_np = self._tensor_to_numpy(input_images)
        pred_np = self._tensor_to_numpy(predictions)
        target_np = self._tensor_to_numpy(targets) if targets is not None else None
        
        # Determine number of columns
        n_cols = 3 if target_np is not None else 2
        n_rows = min(num_samples, input_np.shape[0])
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_rows):
            # Input image
            axes[i, 0].imshow(input_np[i], cmap='gray')
            axes[i, 0].set_title(f'Input {i+1}')
            axes[i, 0].axis('off')
            
            # Prediction
            axes[i, 1].imshow(pred_np[i], cmap='gray')
            axes[i, 1].set_title(f'Prediction {i+1}')
            axes[i, 1].axis('off')
            
            # Target (if available)
            if target_np is not None:
                axes[i, 2].imshow(target_np[i], cmap='gray')
                axes[i, 2].set_title(f'Target {i+1}')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "model_predictions.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_fpss_separation(self,
                           input_images: torch.Tensor,
                           flow_predictions: torch.Tensor,
                           noise_predictions: torch.Tensor,
                           flow_targets: torch.Tensor = None,
                           noise_targets: torch.Tensor = None,
                           num_samples: int = 3,
                           save_path: str = None) -> None:
        """Plot FPSS flow and noise separation results.
        
        Args:
            input_images: Input OCT images
            flow_predictions: Predicted flow components
            noise_predictions: Predicted noise components
            flow_targets: Ground truth flow components (optional)
            noise_targets: Ground truth noise components (optional)
            num_samples: Number of samples to visualize
            save_path: Path to save the plot
        """
        # Convert tensors to numpy
        input_np = self._tensor_to_numpy(input_images)
        flow_pred_np = self._tensor_to_numpy(flow_predictions)
        noise_pred_np = self._tensor_to_numpy(noise_predictions)
        
        flow_target_np = self._tensor_to_numpy(flow_targets) if flow_targets is not None else None
        noise_target_np = self._tensor_to_numpy(noise_targets) if noise_targets is not None else None
        
        # Determine layout
        n_cols = 7 if flow_target_np is not None else 4
        n_rows = min(num_samples, input_np.shape[0])
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_rows):
            col = 0
            
            # Input image
            axes[i, col].imshow(input_np[i], cmap='gray')
            axes[i, col].set_title(f'Input {i+1}')
            axes[i, col].axis('off')
            col += 1
            
            # Flow prediction
            axes[i, col].imshow(flow_pred_np[i], cmap='hot')
            axes[i, col].set_title(f'Flow Pred {i+1}')
            axes[i, col].axis('off')
            col += 1
            
            # Noise prediction
            axes[i, col].imshow(noise_pred_np[i], cmap='gray')
            axes[i, col].set_title(f'Noise Pred {i+1}')
            axes[i, col].axis('off')
            col += 1
            
            # Reconstruction
            reconstruction = flow_pred_np[i] + noise_pred_np[i]
            axes[i, col].imshow(reconstruction, cmap='gray')
            axes[i, col].set_title(f'Reconstruction {i+1}')
            axes[i, col].axis('off')
            col += 1
            
            # Ground truth targets (if available)
            if flow_target_np is not None:
                axes[i, col].imshow(flow_target_np[i], cmap='hot')
                axes[i, col].set_title(f'Flow GT {i+1}')
                axes[i, col].axis('off')
                col += 1
                
                axes[i, col].imshow(noise_target_np[i], cmap='gray')
                axes[i, col].set_title(f'Noise GT {i+1}')
                axes[i, col].axis('off')
                col += 1
                
                # Target reconstruction
                target_reconstruction = flow_target_np[i] + noise_target_np[i]
                axes[i, col].imshow(target_reconstruction, cmap='gray')
                axes[i, col].set_title(f'GT Reconstruction {i+1}')
                axes[i, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "fpss_separation.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_metrics_comparison(self,
                              metrics_dict: Dict[str, List[float]],
                              save_path: str = None) -> None:
        """Plot comparison of different metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            save_path: Path to save the plot
        """
        n_metrics = len(metrics_dict)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            axes[row, col].plot(values, linewidth=2)
            axes[row, col].set_title(metric_name)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_attention_maps(self,
                          input_images: torch.Tensor,
                          attention_maps: torch.Tensor,
                          num_samples: int = 4,
                          save_path: str = None) -> None:
        """Plot attention maps overlay on input images.
        
        Args:
            input_images: Input images tensor
            attention_maps: Attention maps tensor
            num_samples: Number of samples to visualize
            save_path: Path to save the plot
        """
        # Convert tensors to numpy
        input_np = self._tensor_to_numpy(input_images)
        attention_np = self._tensor_to_numpy(attention_maps)
        
        n_rows = min(num_samples, input_np.shape[0])
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_rows):
            # Original image
            axes[i, 0].imshow(input_np[i], cmap='gray')
            axes[i, 0].set_title(f'Input {i+1}')
            axes[i, 0].axis('off')
            
            # Attention map
            axes[i, 1].imshow(attention_np[i], cmap='hot')
            axes[i, 1].set_title(f'Attention {i+1}')
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(input_np[i], cmap='gray')
            axes[i, 2].imshow(attention_np[i], cmap='hot', alpha=0.5)
            axes[i, 2].set_title(f'Overlay {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "attention_maps.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_loss_components(self,
                           loss_dict: Dict[str, List[float]],
                           save_path: str = None) -> None:
        """Plot individual loss components.
        
        Args:
            loss_dict: Dictionary of loss component names and values
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx, (loss_name, values) in enumerate(loss_dict.items()):
            if idx < len(axes):
                axes[idx].plot(values, color=colors[idx % len(colors)], linewidth=2)
                axes[idx].set_title(f'{loss_name} Loss')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel('Loss Value')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(loss_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "loss_components.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_training_report(self,
                             metrics_history: List[Dict[str, float]],
                             training_config: Dict[str, Any],
                             save_path: str = None) -> None:
        """Create comprehensive training report.
        
        Args:
            metrics_history: List of metrics dictionaries
            training_config: Training configuration
            save_path: Path to save the report
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Training curves
        ax1 = plt.subplot(2, 3, 1)
        if metrics_history:
            epochs = range(1, len(metrics_history) + 1)
            if 'train_loss' in metrics_history[0]:
                train_losses = [m['train_loss'] for m in metrics_history]
                ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
            if 'val_loss' in metrics_history[0]:
                val_losses = [m['val_loss'] for m in metrics_history]
                ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: PSNR progression
        ax2 = plt.subplot(2, 3, 2)
        if metrics_history and 'psnr' in metrics_history[0]:
            psnr_values = [m['psnr'] for m in metrics_history]
            ax2.plot(epochs, psnr_values, 'g-', linewidth=2)
            ax2.set_title('PSNR Progression')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('PSNR (dB)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: SSIM progression
        ax3 = plt.subplot(2, 3, 3)
        if metrics_history and 'ssim' in metrics_history[0]:
            ssim_values = [m['ssim'] for m in metrics_history]
            ax3.plot(epochs, ssim_values, 'orange', linewidth=2)
            ax3.set_title('SSIM Progression')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('SSIM')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Configuration summary
        ax4 = plt.subplot(2, 3, (4, 6))
        ax4.axis('off')
        
        # Create text summary
        config_text = "Training Configuration:\n"
        config_text += "-" * 30 + "\n"
        
        key_configs = ['model_name', 'batch_size', 'learning_rate', 'num_epochs', 'loss_function']
        for key in key_configs:
            if key in training_config:
                config_text += f"{key}: {training_config[key]}\n"
        
        if metrics_history:
            config_text += "\nFinal Metrics:\n"
            config_text += "-" * 30 + "\n"
            final_metrics = metrics_history[-1]
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    config_text += f"{key}: {value:.4f}\n"
        
        ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "training_report.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array ready for visualization
        """
        if tensor is None:
            return None
        
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        array = tensor.numpy()
        
        # Handle different tensor shapes
        if array.ndim == 4:  # Batch, Channel, Height, Width
            array = array.squeeze(1)  # Remove channel dimension
        elif array.ndim == 3:  # Channel, Height, Width or Batch, Height, Width
            if array.shape[0] == 1:
                array = array.squeeze(0)  # Remove channel dimension
        
        return array
    
    def save_metrics_json(self, metrics_dict: Dict[str, Any], filename: str = "metrics.json"):
        """Save metrics to JSON file.
        
        Args:
            metrics_dict: Dictionary of metrics
            filename: Name of the JSON file
        """
        with open(self.output_dir / filename, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def close(self):
        """Clean up resources."""
        plt.close('all')