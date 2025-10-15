"""
Configuration Module for IC Logo Detection Training
Centralizes all hyperparameters, paths, and settings for easy modification and debugging
"""

import os
import torch

class TrainingConfig:
    """
    Central configuration class for model training
    All hyperparameters and paths are defined here for easy access and modification
    """
    
    # ===========================
    # PATHS
    # ===========================
    # Base directory of the project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dataset directory containing IC manufacturer logo folders
    DATA_DIR = os.path.join(BASE_DIR, "..", "Dataset")
    
    # Directory to save trained models
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "..", "trained_models")
    
    # Directory to save training logs and checkpoints
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # ===========================
    # MODEL CONFIGURATION
    # ===========================
    # Model architecture (from timm library)
    MODEL_NAME = 'efficientnet_b0'
    
    # Number of IC manufacturer classes
    # Current manufacturers: Infineon, Microchip, NXP, ON Semi, ST Micro, Texas Instruments
    NUM_CLASSES = 6
    
    # Use pretrained weights
    PRETRAINED = True
    
    # ===========================
    # TRAINING HYPERPARAMETERS
    # ===========================
    # Number of training epochs
    EPOCHS = 20
    
    # Batch size for training
    BATCH_SIZE = 16
    
    # Learning rate
    LEARNING_RATE = 1e-4
    
    # Device configuration (CUDA if available, else CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Number of workers for data loading
    NUM_WORKERS = 4
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # ===========================
    # DATA AUGMENTATION SETTINGS
    # ===========================
    # Input image size (EfficientNet expects 224x224)
    IMAGE_SIZE = 224
    
    # Random rotation range (in degrees)
    ROTATION_DEGREES = 10
    
    # Random crop scale range
    CROP_SCALE_MIN = 0.8
    CROP_SCALE_MAX = 1.0
    
    # Color jitter parameters
    BRIGHTNESS_FACTOR = 0.2
    CONTRAST_FACTOR = 0.2
    
    # Normalization values (mean and std for grayscale)
    NORMALIZE_MEAN = [0.5]
    NORMALIZE_STD = [0.5]
    
    # ===========================
    # MODEL CHECKPOINT SETTINGS
    # ===========================
    # Save model every N epochs
    SAVE_FREQUENCY = 5
    
    # Final model filename
    FINAL_MODEL_NAME = "logo_classifier.pth"
    
    # Best model filename (based on accuracy)
    BEST_MODEL_NAME = "logo_classifier_best.pth"
    
    # ===========================
    # VALIDATION SETTINGS
    # ===========================
    # Train/validation split ratio (if validation needed)
    VALIDATION_SPLIT = 0.2
    
    # Use validation during training
    USE_VALIDATION = False
    
    @classmethod
    def print_config(cls):
        """
        Print all configuration settings for debugging
        """
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Model Save Directory: {cls.MODEL_SAVE_DIR}")
        print("=" * 60)
    
    @classmethod
    def create_directories(cls):
        """
        Create necessary directories if they don't exist
        """
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        print(f"✓ Created directory: {cls.MODEL_SAVE_DIR}")
        print(f"✓ Created directory: {cls.LOG_DIR}")


# Create a singleton instance
config = TrainingConfig()
