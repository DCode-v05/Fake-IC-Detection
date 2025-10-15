"""
Main Training Script for IC Logo Detection
Orchestrates all modules to train the EfficientNet model
Run this script to start training
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from config.training_config import config
from utils.data_loader import DataPreprocessor
from models.efficientnet_model import LogoClassifier
from utils.trainer import Trainer


def main():
    """
    Main training pipeline
    """
    print("\n" + "=" * 60)
    print("IC LOGO DETECTION - TRAINING PIPELINE")
    print("=" * 60 + "\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    print(f"✓ Random seed set to: {config.RANDOM_SEED}\n")
    
    # Create necessary directories
    config.create_directories()
    print()
    
    # Print configuration
    config.print_config()
    print()
    
    # ===========================
    # 1. DATA PREPARATION
    # ===========================
    print("STEP 1: Preparing Dataset")
    print("-" * 60)
    
    preprocessor = DataPreprocessor(config)
    dataset = preprocessor.load_dataset()
    train_loader = preprocessor.create_dataloaders(dataset)
    
    # Get class names
    class_names = preprocessor.get_class_names(dataset)
    print(f"\nDetected IC Manufacturers: {class_names}")
    print()
    
    # ===========================
    # 2. MODEL CREATION
    # ===========================
    print("STEP 2: Creating Model")
    print("-" * 60)
    
    classifier = LogoClassifier(config)
    model = classifier.create_model()
    model = classifier.move_to_device(model)
    classifier.get_model_summary(model)
    
    # ===========================
    # 3. TRAINING
    # ===========================
    print("STEP 3: Training Model")
    print("-" * 60)
    
    trainer = Trainer(model, train_loader, config)
    train_losses, train_accuracies = trainer.train()
    
    # ===========================
    # 4. SAVE TRAINING HISTORY
    # ===========================
    print("STEP 4: Saving Training History")
    print("-" * 60)
    
    history = trainer.get_training_history()
    history_path = os.path.join(config.MODEL_SAVE_DIR, "training_history.txt")
    
    with open(history_path, 'w') as f:
        f.write("IC Logo Detection - Training History\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config.MODEL_NAME}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Total Epochs: {config.EPOCHS}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n\n")
        f.write("Epoch-wise Performance:\n")
        f.write("-" * 60 + "\n")
        
        for epoch in range(len(train_losses)):
            f.write(f"Epoch {epoch+1:2d}: ")
            f.write(f"Loss={train_losses[epoch]:.4f}, ")
            f.write(f"Accuracy={train_accuracies[epoch]:.2f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Best Training Accuracy: {trainer.best_accuracy:.2f}%\n")
    
    print(f"✓ Training history saved to: {history_path}\n")
    
    # ===========================
    # 5. FINAL SUMMARY
    # ===========================
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTrained Model Location:")
    print(f"  → {os.path.join(config.MODEL_SAVE_DIR, config.FINAL_MODEL_NAME)}")
    print(f"\nBest Model Location:")
    print(f"  → {os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)}")
    print(f"\nFinal Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Best Training Accuracy: {trainer.best_accuracy:.2f}%")
    print("\n" + "=" * 60 + "\n")
    
    return model, trainer


if __name__ == "__main__":
    try:
        model, trainer = main()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Training failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
