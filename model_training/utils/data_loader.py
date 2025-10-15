"""
Data Preprocessing Module
Handles data transformations, augmentation, and DataLoader creation
Modular design for easy debugging and modification
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config.training_config import config


class DataPreprocessor:
    """
    Handles all data preprocessing operations including:
    - Image transformations and augmentation
    - Dataset loading
    - DataLoader creation
    """
    
    def __init__(self, config_obj=config):
        """
        Initialize the data preprocessor with configuration
        
        Args:
            config_obj: Configuration object containing all settings
        """
        self.config = config_obj
        self.transform = self._create_transforms()
        
    def _create_transforms(self):
        """
        Create image transformation pipeline for IC logo images
        Optimized for grayscale IC markings with augmentation
        
        Returns:
            torchvision.transforms.Compose: Transformation pipeline
        """
        transform_pipeline = transforms.Compose([
            # Convert to 3-channel grayscale (required for EfficientNet)
            transforms.Grayscale(num_output_channels=3),
            
            # Random rotation for varied IC orientations
            transforms.RandomRotation(self.config.ROTATION_DEGREES),
            
            # Random resized crop for scale invariance
            transforms.RandomResizedCrop(
                self.config.IMAGE_SIZE,
                scale=(self.config.CROP_SCALE_MIN, self.config.CROP_SCALE_MAX)
            ),
            
            # Random horizontal flip for data augmentation
            transforms.RandomHorizontalFlip(),
            
            # Color jitter for lighting variations
            transforms.ColorJitter(
                brightness=self.config.BRIGHTNESS_FACTOR,
                contrast=self.config.CONTRAST_FACTOR
            ),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Normalize pixel values
            transforms.Normalize(
                self.config.NORMALIZE_MEAN,
                self.config.NORMALIZE_STD
            )
        ])
        
        print("✓ Created transformation pipeline")
        return transform_pipeline
    
    def create_validation_transforms(self):
        """
        Create validation transformation pipeline (no augmentation)
        
        Returns:
            torchvision.transforms.Compose: Validation transformation pipeline
        """
        val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.config.IMAGE_SIZE),
            transforms.CenterCrop(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                self.config.NORMALIZE_MEAN,
                self.config.NORMALIZE_STD
            )
        ])
        
        return val_transform
    
    def load_dataset(self):
        """
        Load the IC logo dataset from directory
        
        Returns:
            datasets.ImageFolder: Loaded dataset
        """
        try:
            dataset = datasets.ImageFolder(
                self.config.DATA_DIR,
                transform=self.transform
            )
            
            print(f"✓ Loaded dataset from: {self.config.DATA_DIR}")
            print(f"  Total images: {len(dataset)}")
            print(f"  Number of classes: {len(dataset.classes)}")
            print(f"  Classes: {dataset.classes}")
            
            return dataset
            
        except FileNotFoundError:
            print(f"✗ Error: Dataset directory not found at {self.config.DATA_DIR}")
            raise
        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            raise
    
    def create_dataloaders(self, dataset, use_validation=None):
        """
        Create training (and optionally validation) DataLoaders
        
        Args:
            dataset: The loaded dataset
            use_validation: Whether to create validation split
            
        Returns:
            DataLoader or tuple: Train loader (and val loader if use_validation=True)
        """
        use_validation = use_validation or self.config.USE_VALIDATION
        
        if use_validation:
            # Split dataset into train and validation
            total_size = len(dataset)
            val_size = int(total_size * self.config.VALIDATION_SPLIT)
            train_size = total_size - val_size
            
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config.RANDOM_SEED)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True if self.config.DEVICE == "cuda" else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True if self.config.DEVICE == "cuda" else False
            )
            
            print(f"✓ Created DataLoaders:")
            print(f"  Training samples: {train_size}")
            print(f"  Validation samples: {val_size}")
            print(f"  Batch size: {self.config.BATCH_SIZE}")
            
            return train_loader, val_loader
        
        else:
            # Create only training DataLoader
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True if self.config.DEVICE == "cuda" else False
            )
            
            print(f"✓ Created Training DataLoader:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Batch size: {self.config.BATCH_SIZE}")
            print(f"  Number of batches: {len(train_loader)}")
            
            return train_loader
    
    def get_class_names(self, dataset):
        """
        Get the class names from the dataset
        
        Args:
            dataset: The loaded dataset
            
        Returns:
            list: List of class names
        """
        return dataset.classes


def create_data_pipeline(config_obj=config):
    """
    Convenience function to create complete data pipeline
    
    Args:
        config_obj: Configuration object
        
    Returns:
        tuple: (preprocessor, dataset, dataloader)
    """
    preprocessor = DataPreprocessor(config_obj)
    dataset = preprocessor.load_dataset()
    dataloader = preprocessor.create_dataloaders(dataset)
    
    return preprocessor, dataset, dataloader
