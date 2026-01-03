"""
Unified data loading module for FPSS framework.

This module provides a consistent interface for loading different types of OCT datasets
with standardized preprocessing and augmentation pipelines.
"""

import os
import glob
import re
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class OCTDataset(Dataset, ABC):
    """Abstract base class for OCT datasets."""
    
    def __init__(self, 
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 verbose: bool = False):
        """Initialize OCT dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to apply to the data
            target_size: Target size for images (H, W)
            normalize: Whether to normalize images to [0, 1]
            verbose: Whether to print verbose information
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.verbose = verbose
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        
        if self.verbose:
            print(f"Loaded {len(self.data_paths)} samples from {root_dir}")
    
    @abstractmethod
    def _load_data_paths(self) -> List[str]:
        """Load data paths specific to the dataset format.
        
        Returns:
            List of data paths
        """
        pass
    
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the data item
        """
        # Load raw data
        data = self._load_raw_data(idx)
        
        # Apply preprocessing
        data = self._preprocess_data(data)
        
        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)
        
        return data
    
    @abstractmethod
    def _load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load raw data for a specific index.
        
        Args:
            idx: Index of the data to load
            
        Returns:
            Dictionary containing raw data
        """
        pass
    
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Apply standard preprocessing to the data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Preprocessed data dictionary with torch tensors
        """
        processed = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert to float32
                value = value.astype(np.float32)
                
                # Normalize if requested
                if self.normalize and value.max() > 1.0:
                    value = value / 255.0
                
                # Resize if needed
                if value.shape[-2:] != self.target_size:
                    value = self._resize_image(value, self.target_size)
                
                # Convert to tensor
                if value.ndim == 2:
                    value = value[None, ...]  # Add channel dimension
                elif value.ndim == 3 and value.shape[0] != 1:
                    value = value[None, ...]  # Add batch dimension
                
                processed[key] = torch.from_numpy(value)
            else:
                processed[key] = value
        
        return processed
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (H, W)
            
        Returns:
            Resized image
        """
        from skimage.transform import resize
        
        if image.ndim == 2:
            return resize(image, target_size, preserve_range=True, anti_aliasing=True)
        elif image.ndim == 3:
            return resize(image, target_size, preserve_range=True, anti_aliasing=True)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")


class SDOCTDataset(OCTDataset):
    """Dataset for SD-OCT (Spectral Domain OCT) data."""
    
    def __init__(self, 
                 root_dir: str,
                 n_patients: int = 10,
                 start_patient: int = 1,
                 n_images: int = 10,
                 **kwargs):
        """Initialize SD-OCT dataset.
        
        Args:
            root_dir: Root directory containing patient folders
            n_patients: Number of patients to include
            start_patient: Starting patient index
            n_images: Number of images per patient
            **kwargs: Additional arguments for OCTDataset
        """
        self.n_patients = n_patients
        self.start_patient = start_patient
        self.n_images = n_images
        
        super().__init__(root_dir, **kwargs)
    
    def _load_data_paths(self) -> List[str]:
        """Load SD-OCT data paths.
        
        Returns:
            List of data paths
        """
        data_paths = []
        
        # Iterate through patient types (0: non-diabetic, 1: diabetic type 1, 2: diabetic type 2)
        for patient_type in [0, 1, 2]:
            type_dir = os.path.join(self.root_dir, str(patient_type))
            if not os.path.exists(type_dir):
                continue
            
            # Get all patient folders
            patient_folders = sorted([f for f in os.listdir(type_dir) 
                                    if os.path.isdir(os.path.join(type_dir, f))])
            
            # Select subset of patients
            start_idx = self.start_patient - 1
            end_idx = start_idx + self.n_patients
            selected_patients = patient_folders[start_idx:end_idx]
            
            for patient_folder in selected_patients:
                patient_path = os.path.join(type_dir, patient_folder)
                
                # Find image files
                image_files = self._find_image_files(patient_path)
                
                # Select subset of images
                if len(image_files) > self.n_images:
                    image_files = image_files[:self.n_images]
                
                data_paths.extend(image_files)
        
        return data_paths
    
    def _find_image_files(self, patient_path: str) -> List[str]:
        """Find image files in patient directory.
        
        Args:
            patient_path: Path to patient directory
            
        Returns:
            List of image file paths
        """
        extensions = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"]
        image_files = []
        
        for ext in extensions:
            files = glob.glob(os.path.join(patient_path, ext))
            image_files.extend(files)
        
        # Sort files by number in filename (if present)
        def extract_number(filename):
            match = re.search(r'\((\d+)\)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        return sorted(image_files, key=extract_number)
    
    def _load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load raw SD-OCT data.
        
        Args:
            idx: Index of the data to load
            
        Returns:
            Dictionary containing raw data
        """
        image_path = self.data_paths[idx]
        
        try:
            # Load image
            image = io.imread(image_path)
            
            # Convert to grayscale if needed
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            
            return {
                'image': image,
                'path': image_path,
                'patient_id': self._extract_patient_id(image_path)
            }
        
        except Exception as e:
            if self.verbose:
                print(f"Error loading {image_path}: {e}")
            
            # Return empty image
            return {
                'image': np.zeros(self.target_size, dtype=np.float32),
                'path': image_path,
                'patient_id': 'unknown'
            }
    
    def _extract_patient_id(self, image_path: str) -> str:
        """Extract patient ID from image path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Patient ID string
        """
        # Extract patient ID from path structure
        path_parts = image_path.split(os.sep)
        if len(path_parts) >= 2:
            return path_parts[-2]  # Patient folder name
        return 'unknown'


class PairedOCTDataset(OCTDataset):
    """Dataset for paired OCT data (e.g., for speckle separation tasks)."""
    
    def __init__(self, 
                 root_dir: str,
                 sequence_length: int = 3,
                 create_pairs: bool = True,
                 **kwargs):
        """Initialize paired OCT dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            sequence_length: Length of image sequences for pairing
            create_pairs: Whether to create OCT-OCTA pairs
            **kwargs: Additional arguments for OCTDataset
        """
        self.sequence_length = sequence_length
        self.create_pairs = create_pairs
        
        super().__init__(root_dir, **kwargs)
    
    def _load_data_paths(self) -> List[str]:
        """Load paired OCT data paths.
        
        Returns:
            List of data paths
        """
        # For now, use the same structure as SD-OCT
        # This can be extended for specific paired data formats
        return self._load_sdoct_paths()
    
    def _load_sdoct_paths(self) -> List[str]:
        """Load SD-OCT style paths.
        
        Returns:
            List of data paths
        """
        data_paths = []
        
        for patient_type in [0, 1, 2]:
            type_dir = os.path.join(self.root_dir, str(patient_type))
            if not os.path.exists(type_dir):
                continue
            
            patient_folders = sorted([f for f in os.listdir(type_dir) 
                                    if os.path.isdir(os.path.join(type_dir, f))])
            
            for patient_folder in patient_folders:
                patient_path = os.path.join(type_dir, patient_folder)
                image_files = self._find_image_files(patient_path)
                
                # Create sequences for pairing
                if self.create_pairs and len(image_files) >= self.sequence_length:
                    for i in range(len(image_files) - self.sequence_length + 1):
                        sequence = image_files[i:i + self.sequence_length]
                        data_paths.append(sequence)
                else:
                    data_paths.extend(image_files)
        
        return data_paths
    
    def _find_image_files(self, patient_path: str) -> List[str]:
        """Find image files in patient directory."""
        extensions = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"]
        image_files = []
        
        for ext in extensions:
            files = glob.glob(os.path.join(patient_path, ext))
            image_files.extend(files)
        
        def extract_number(filename):
            match = re.search(r'\((\d+)\)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        return sorted(image_files, key=extract_number)
    
    def _load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load raw paired OCT data.
        
        Args:
            idx: Index of the data to load
            
        Returns:
            Dictionary containing raw data
        """
        data_path = self.data_paths[idx]
        
        if isinstance(data_path, list):
            # Load sequence of images
            images = []
            for img_path in data_path:
                try:
                    img = io.imread(img_path)
                    if img.ndim == 3:
                        img = np.mean(img, axis=2)
                    images.append(img)
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading {img_path}: {e}")
                    images.append(np.zeros(self.target_size, dtype=np.float32))
            
            # Create OCT-OCTA pair if requested
            if self.create_pairs and len(images) >= 2:
                oct_image = images[1]  # Middle image as OCT
                octa_image = self._create_octa_from_sequence(images)
                
                return {
                    'oct': oct_image,
                    'octa': octa_image,
                    'path': data_path[1],
                    'patient_id': self._extract_patient_id(data_path[1])
                }
            else:
                return {
                    'image': images[0] if images else np.zeros(self.target_size, dtype=np.float32),
                    'path': data_path[0] if data_path else 'unknown',
                    'patient_id': self._extract_patient_id(data_path[0] if data_path else 'unknown')
                }
        else:
            # Single image
            try:
                image = io.imread(data_path)
                if image.ndim == 3:
                    image = np.mean(image, axis=2)
                
                return {
                    'image': image,
                    'path': data_path,
                    'patient_id': self._extract_patient_id(data_path)
                }
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {data_path}: {e}")
                
                return {
                    'image': np.zeros(self.target_size, dtype=np.float32),
                    'path': data_path,
                    'patient_id': 'unknown'
                }
    
    def _create_octa_from_sequence(self, images: List[np.ndarray]) -> np.ndarray:
        """Create OCTA image from sequence using temporal variance.
        
        Args:
            images: List of sequential OCT images
            
        Returns:
            OCTA image
        """
        if len(images) < 2:
            return images[0] if images else np.zeros(self.target_size, dtype=np.float32)
        
        # Stack images and compute temporal variance
        image_stack = np.stack(images, axis=0)
        octa = np.var(image_stack, axis=0)
        
        # Normalize OCTA
        if octa.max() > 0:
            octa = (octa - octa.min()) / (octa.max() - octa.min())
        
        return octa
    
    def _extract_patient_id(self, image_path: str) -> str:
        """Extract patient ID from image path."""
        if isinstance(image_path, list):
            image_path = image_path[0]
        
        path_parts = image_path.split(os.sep)
        if len(path_parts) >= 2:
            return path_parts[-2]
        return 'unknown'


class DataLoaderFactory:
    """Factory for creating data loaders with consistent configurations."""
    
    @staticmethod
    def create_dataloader(dataset_type: str,
                         root_dir: str,
                         batch_size: int = 4,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         **dataset_kwargs) -> DataLoader:
        """Create a data loader for the specified dataset type.
        
        Args:
            dataset_type: Type of dataset ('sdoct', 'paired')
            root_dir: Root directory containing the dataset
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            **dataset_kwargs: Additional arguments for the dataset
            
        Returns:
            DataLoader instance
            
        Raises:
            ValueError: If dataset type is not recognized
        """
        dataset_classes = {
            'sdoct': SDOCTDataset,
            'paired': PairedOCTDataset
        }
        
        if dataset_type not in dataset_classes:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Available: {list(dataset_classes.keys())}")
        
        # Create dataset
        dataset_class = dataset_classes[dataset_type]
        dataset = dataset_class(root_dir, **dataset_kwargs)
        
        # Create data loader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    @staticmethod
    def create_train_val_loaders(dataset_type: str,
                               root_dir: str,
                               train_ratio: float = 0.8,
                               batch_size: int = 4,
                               num_workers: int = 4,
                               **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders.
        
        Args:
            dataset_type: Type of dataset
            root_dir: Root directory containing the dataset
            train_ratio: Ratio of data to use for training
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            **dataset_kwargs: Additional arguments for the dataset
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create full dataset
        dataset_classes = {
            'sdoct': SDOCTDataset,
            'paired': PairedOCTDataset
        }
        
        if dataset_type not in dataset_classes:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        dataset_class = dataset_classes[dataset_type]
        full_dataset = dataset_class(root_dir, **dataset_kwargs)
        
        # Split dataset
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader