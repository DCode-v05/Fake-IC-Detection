"""
Training Module
Handles the complete training loop with progress tracking and checkpointing
Modular design for easy debugging and monitoring
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config.training_config import config


class Trainer:
    """
    Handles the training process for the logo classifier
    Includes training loop, loss tracking, and model checkpointing
    """
    
    def __init__(self, model, train_loader, config_obj=config, val_loader=None):
        """
        Initialize the trainer
        
        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            config_obj: Configuration object
            val_loader: DataLoader for validation data (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config_obj
        
        # Training components
        self.criterion = None
        self.optimizer = None
        
        # Tracking metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        
    def setup_training(self):
        """
        Setup loss function and optimizer
        """
        # Loss function (Cross Entropy for multi-class classification)
        self.criterion = nn.CrossEntropyLoss()
        print("✓ Loss Function: CrossEntropyLoss")
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        print(f"✓ Optimizer: Adam (lr={self.config.LEARNING_RATE})")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        loop = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch+1}/{self.config.EPOCHS}]",
            leave=True
        )
        
        for batch_idx, (images, labels) in enumerate(loop):
            # Move data to device
            images = images.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            loop.set_postfix(
                loss=loss.item(),
                acc=current_acc
            )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Regular checkpoint
        if (epoch + 1) % self.config.SAVE_FREQUENCY == 0:
            checkpoint_path = os.path.join(
                self.config.MODEL_SAVE_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses[-1],
                'train_acc': self.train_accuracies[-1],
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Best model checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.MODEL_SAVE_DIR,
                self.config.BEST_MODEL_NAME
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses[-1],
                'train_acc': self.train_accuracies[-1],
            }, best_path)
            print(f"  ✓ Best model saved: {best_path}")
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        # Setup training components
        self.setup_training()
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate if validation loader provided
            if self.val_loader is not None:
                val_loss, val_acc = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                print(f"\nEpoch [{epoch+1}/{self.config.EPOCHS}] Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                # Check if best model
                is_best = val_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_acc
                    print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")
            else:
                print(f"\nEpoch [{epoch+1}/{self.config.EPOCHS}] Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                
                # Check if best model (based on training accuracy)
                is_best = train_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = train_acc
                    print(f"  ✓ New best training accuracy: {train_acc:.2f}%")
            
            # Save checkpoints
            self.save_checkpoint(epoch, is_best)
            print()
        
        # Save final model
        final_path = os.path.join(
            self.config.MODEL_SAVE_DIR,
            self.config.FINAL_MODEL_NAME
        )
        torch.save(self.model.state_dict(), final_path)
        
        print("=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Final Model saved to: {final_path}")
        print(f"Best Accuracy: {self.best_accuracy:.2f}%")
        print("=" * 60 + "\n")
        
        return self.train_losses, self.train_accuracies
    
    def get_training_history(self):
        """
        Get training history metrics
        
        Returns:
            dict: Dictionary containing training metrics
        """
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
        }
        
        if self.val_loader is not None:
            history['val_losses'] = self.val_losses
            history['val_accuracies'] = self.val_accuracies
        
        return history
