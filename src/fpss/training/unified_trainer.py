"""
Unified training framework for FPSS models.

This module provides a consistent training interface for all FPSS models
with configurable loss functions, optimizers, and training strategies.
"""

import os
import time
import yaml
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ..models.model_factory import ModelFactory, BaseModel
from ..losses.unified_losses import get_loss_function
from ..data.unified_dataloader import DataLoaderFactory
from ..utils.metrics import MetricsCalculator
from ..utils.visualization import TrainingVisualizer


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Model configuration
    model_name: str = "fpss_attention"
    model_config: Dict[str, Any] = None
    
    # Data configuration
    dataset_type: str = "paired"
    dataset_root: str = ""
    n_patients: int = 10
    start_patient: int = 1
    n_images: int = 10
    batch_size: int = 4
    num_workers: int = 4
    train_ratio: float = 0.8
    
    # Training configuration
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    scheduler_params: Dict[str, Any] = None
    
    # Loss configuration
    loss_function: str = "fpss"
    loss_params: Dict[str, Any] = None
    
    # Checkpoint configuration
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    load_checkpoint: Optional[str] = None
    
    # Logging configuration
    log_dir: str = "logs"
    log_frequency: int = 10
    visualize: bool = True
    
    # Device configuration
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Validation configuration
    validation_frequency: int = 5
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.model_config is None:
            self.model_config = {}
        if self.loss_params is None:
            self.loss_params = {}
        if self.scheduler_params is None:
            self.scheduler_params = {}


class UnifiedTrainer:
    """Unified trainer for FPSS models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the unified trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.metrics_calculator = None
        self.visualizer = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Setup training
        self._setup_training()
    
    def _setup_training(self):
        """Setup all training components."""
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Initialize model
        self.model = ModelFactory.create_model(
            self.config.model_name,
            self.config.model_config
        )
        self.model.to(self.device)
        
        # Initialize data loaders
        self.train_loader, self.val_loader = DataLoaderFactory.create_train_val_loaders(
            dataset_type=self.config.dataset_type,
            root_dir=self.config.dataset_root,
            train_ratio=self.config.train_ratio,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            n_patients=self.config.n_patients,
            start_patient=self.config.start_patient,
            n_images=self.config.n_images,
            verbose=True
        )
        
        # Initialize loss function
        self.loss_function = get_loss_function(
            self.config.loss_function,
            **self.config.loss_params
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize logging
        self.writer = SummaryWriter(self.config.log_dir)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize visualizer
        if self.config.visualize:
            self.visualizer = TrainingVisualizer(self.config.log_dir)
        
        # Load checkpoint if specified
        if self.config.load_checkpoint:
            self._load_checkpoint(self.config.load_checkpoint)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration.
        
        Returns:
            Optimizer instance
        """
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration.
        
        Returns:
            Scheduler instance or None
        """
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                **self.config.scheduler_params
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get("step_size", 30),
                gamma=self.config.scheduler_params.get("gamma", 0.1)
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_params.get("factor", 0.5),
                patience=self.config.scheduler_params.get("patience", 10)
            )
        elif self.config.scheduler.lower() == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Dataset: {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val")
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_loss = self._train_epoch()
                
                # Validate
                val_loss = None
                if epoch % self.config.validation_frequency == 0:
                    val_loss = self._validate_epoch()
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss is not None:
                            self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Log progress
                self._log_epoch(epoch, train_loss, val_loss)
                
                # Save checkpoint
                if epoch % self.config.save_frequency == 0:
                    self._save_checkpoint(epoch, is_best=False)
                
                # Check for best model
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping
                if (self.early_stopping_counter >= self.config.early_stopping_patience and
                    self.config.early_stopping_patience > 0):
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        
        finally:
            # Save final checkpoint
            self._save_checkpoint(self.current_epoch, is_best=False)
            
            # Close writer
            if self.writer:
                self.writer.close()
            
            print("Training completed!")
    
    def _train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                loss = self._compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch progress
            if batch_idx % self.config.log_frequency == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    loss = self._compute_loss(batch)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # This is a simplified version - actual implementation depends on the specific loss function
        if self.config.loss_function == "fpss":
            # For FPSS loss, we need flow and noise components
            if 'oct' in batch and 'octa' in batch:
                input_image = batch['oct']
                target_flow = batch['octa']
                target_noise = input_image - target_flow
                
                # Forward pass through model
                output = self.model(input_image)
                
                # Extract flow and noise predictions
                if isinstance(output, tuple) and len(output) == 2:
                    flow_pred, noise_pred = output
                else:
                    # Assume single output, split into flow and noise
                    flow_pred = output
                    noise_pred = input_image - flow_pred
                
                # Compute FPSS loss
                loss, loss_dict = self.loss_function(
                    flow_pred, noise_pred, target_flow, target_noise, input_image
                )
                
                return loss
            else:
                # Fallback to simple MSE
                input_image = batch['image']
                output = self.model(input_image)
                return nn.MSELoss()(output, input_image)
        else:
            # For other loss functions
            input_image = batch['image']
            output = self.model(input_image)
            return self.loss_function(output, input_image)
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """Log epoch results.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss (optional)
        """
        # Update training history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        if val_loss is not None:
            self.training_history['val_loss'].append(val_loss)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
        print(f"Epoch {epoch + 1}/{self.config.num_epochs}, "
              f"Train Loss: {train_loss:.4f}{val_str}, "
              f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0.0,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.config.model_name}_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                f"{self.config.model_name}_best.pth"
            )
            torch.save(checkpoint, best_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        })
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def load_config_from_yaml(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Training configuration
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TrainingConfig(**config_dict)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FPSS model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    
    # Create trainer and start training
    trainer = UnifiedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()