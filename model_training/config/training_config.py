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
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DATA_DIR = os.path.join(BASE_DIR, "..", "Dataset")
    
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "..", "trained_models")
    
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    MODEL_NAME = 'efficientnet_b0'
    
    NUM_CLASSES = 6
    
    PRETRAINED = True
    
    EPOCHS = 20
    
    BATCH_SIZE = 16
    
    LEARNING_RATE = 1e-4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    NUM_WORKERS = 4
    
    RANDOM_SEED = 42
    
    IMAGE_SIZE = 224
    
    ROTATION_DEGREES = 10
    
    CROP_SCALE_MIN = 0.8
    CROP_SCALE_MAX = 1.0
    
    BRIGHTNESS_FACTOR = 0.2
    CONTRAST_FACTOR = 0.2
    
    NORMALIZE_MEAN = [0.5]
    NORMALIZE_STD = [0.5]
    
    SAVE_FREQUENCY = 5
    
    FINAL_MODEL_NAME = "logo_classifier.pth"
    
    BEST_MODEL_NAME = "logo_classifier_best.pth"
    
    VALIDATION_SPLIT = 0.2
    
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


config = TrainingConfig()
