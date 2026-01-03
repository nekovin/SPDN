#!/usr/bin/env python3
"""
Unified FPSS training script.

This script provides a clean, simple interface for training FPSS models
using the unified framework with configuration files.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpss import UnifiedTrainer, TrainingConfig, load_config_from_yaml


def create_default_config() -> TrainingConfig:
    """Create default training configuration.
    
    Returns:
        Default training configuration
    """
    return TrainingConfig(
        # Model configuration
        model_name="fpss_attention",
        model_config={
            "n_channels": 1,
            "n_classes": 1,
            "initial_features": 32,
            "depth": 4,
            "block_depth": 3,
            "bilinear": True
        },
        
        # Data configuration
        dataset_type="paired",
        dataset_root=os.getenv("DATASET_DIR_PATH", "/path/to/dataset"),
        n_patients=10,
        start_patient=1,
        n_images=10,
        batch_size=4,
        num_workers=4,
        train_ratio=0.8,
        
        # Training configuration
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer="adam",
        scheduler="cosine",
        
        # Loss configuration
        loss_function="fpss",
        loss_params={
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "delta": 1.0,
            "use_content_loss": True,
            "use_edge_loss": True
        },
        
        # Checkpoint configuration
        checkpoint_dir="checkpoints",
        save_frequency=10,
        
        # Logging configuration
        log_dir="logs",
        log_frequency=10,
        visualize=True,
        
        # Device configuration
        device="cuda",
        mixed_precision=True,
        
        # Validation configuration
        validation_frequency=5,
        early_stopping_patience=20
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train FPSS model with unified framework")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="fpss_attention",
        choices=["fpss_attention", "fpss_attention_small", "fpss_no_attention"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset-root", 
        type=str,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--resume", 
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--no-mixed-precision", 
        action="store_true",
        help="Disable mixed precision training"
    )
    parser.add_argument(
        "--no-visualization", 
        action="store_true",
        help="Disable training visualizations"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.dataset_root:
        config.dataset_root = args.dataset_root
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.resume:
        config.load_checkpoint = args.resume
    if args.device:
        config.device = args.device
    if args.no_mixed_precision:
        config.mixed_precision = False
    if args.no_visualization:
        config.visualize = False
    
    # Validate dataset path
    if not config.dataset_root or not os.path.exists(config.dataset_root):
        print(f"Error: Dataset directory '{config.dataset_root}' does not exist.")
        print("Please set DATASET_DIR_PATH environment variable or use --dataset-root argument.")
        sys.exit(1)
    
    # Print configuration summary
    print("\n" + "="*50)
    print("FPSS Training Configuration")
    print("="*50)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_root}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Log dir: {config.log_dir}")
    if config.load_checkpoint:
        print(f"Resume from: {config.load_checkpoint}")
    print("="*50 + "\n")
    
    # Create trainer and start training
    try:
        trainer = UnifiedTrainer(config)
        trainer.train()
        
        print("\nTraining completed successfully!")
        print(f"Checkpoints saved in: {config.checkpoint_dir}")
        print(f"Logs saved in: {config.log_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()