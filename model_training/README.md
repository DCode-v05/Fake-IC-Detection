# IC Logo Detection - Model Training

Modular training pipeline for detecting IC manufacturer logos using EfficientNet.

## Overview

This module trains a deep learning model to classify IC (Integrated Circuit) manufacturer logos from images. The trained model will be used in the AOI (Automated Optical Inspection) system for identifying genuine vs. fake IC components.

## Supported Manufacturers

The model is trained to recognize logos from:
- **Infineon**
- **Microchip**
- **NXP Semiconductors**
- **ON Semiconductor**
- **STMicroelectronics**
- **Texas Instruments**

## Project Structure

```
model_training/
├── config/
│   └── training_config.py      # Centralized configuration (hyperparameters, paths)
├── models/
│   └── efficientnet_model.py   # EfficientNet model architecture and utilities
├── utils/
│   ├── data_loader.py          # Data preprocessing and augmentation
│   └── trainer.py              # Training loop and checkpoint management
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Navigate to the model_training directory:
```cmd
cd "d:\Deni\Mr. Tech\AI\Projects\Fake IC Marking Detection\model_training"
```

2. Install required dependencies:
```cmd
pip install -r requirements.txt
```

**Note:** For GPU support, install PyTorch with CUDA:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Quick Start

Run the training script:
```cmd
python train.py
```

The script will:
1. Load the dataset from the `Dataset` folder
2. Create the EfficientNet-B0 model
3. Train for 20 epochs (default)
4. Save the trained model to `trained_models/` directory

### Configuration

All training parameters can be modified in `config/training_config.py`:

```python
# Key parameters
EPOCHS = 20              # Number of training epochs
BATCH_SIZE = 16          # Batch size
LEARNING_RATE = 1e-4     # Learning rate
MODEL_NAME = 'efficientnet_b0'  # Model architecture
NUM_CLASSES = 6          # Number of logo classes
```

### Model Architecture

- **Base Model:** EfficientNet-B0 (pretrained on ImageNet)
- **Custom Head:** Fully connected layer for 6-class classification
- **Input Size:** 224x224 RGB images
- **Total Parameters:** ~5.3M (trainable)

### Data Augmentation

The training pipeline includes:
- Grayscale conversion (3-channel for model compatibility)
- Random rotation (±10 degrees)
- Random resized crop (80-100% scale)
- Random horizontal flip
- Color jitter (brightness & contrast)
- Normalization

## Output Files

After training, the following files are generated:

### Trained Models Directory (`../trained_models/`)
- `logo_classifier.pth` - Final trained model
- `logo_classifier_best.pth` - Best model (highest accuracy)
- `checkpoint_epoch_X.pth` - Periodic checkpoints (every 5 epochs)
- `training_history.txt` - Epoch-wise training metrics

## Module Details

### 1. Configuration Module (`config/training_config.py`)
- Centralizes all hyperparameters
- Easy modification without code changes
- Automatic directory creation
- Device detection (CPU/GPU)

### 2. Data Loader Module (`utils/data_loader.py`)
- Image transformation pipeline
- Dataset loading from folders
- DataLoader creation with batching
- Support for train/validation split

### 3. Model Module (`models/efficientnet_model.py`)
- EfficientNet model creation
- Device management
- Parameter counting
- Model save/load utilities

### 4. Trainer Module (`utils/trainer.py`)
- Complete training loop
- Progress tracking with tqdm
- Automatic checkpointing
- Best model selection
- Training history logging

### 5. Main Script (`train.py`)
- Orchestrates all modules
- End-to-end training pipeline
- Error handling
- Results summary

## Modular Design Benefits

✅ **Easy Debugging:** Each component is isolated and testable  
✅ **Customizable:** Modify any module without affecting others  
✅ **Maintainable:** Clear separation of concerns  
✅ **Extensible:** Easy to add new features (e.g., validation, learning rate scheduling)

## Training Tips

### For Better Accuracy:
1. Increase `EPOCHS` (e.g., 50-100)
2. Use data augmentation (already enabled)
3. Add validation split (`USE_VALIDATION = True`)
4. Experiment with learning rate

### For Faster Training:
1. Increase `BATCH_SIZE` (if GPU memory allows)
2. Use GPU (CUDA)
3. Reduce image size (modify `IMAGE_SIZE`)

### For GPU Memory Issues:
1. Reduce `BATCH_SIZE`
2. Use smaller model (`efficientnet_b0` → `mobilenetv3_small_100`)

## Expected Performance

With default settings:
- **Training Time:** ~5-10 minutes (GPU) / ~30-60 minutes (CPU)
- **Expected Accuracy:** >95% (depends on dataset quality)
- **Model Size:** ~21 MB

## Troubleshooting

**Error: CUDA out of memory**
- Solution: Reduce `BATCH_SIZE` in config

**Error: Dataset directory not found**
- Solution: Ensure `Dataset` folder exists with manufacturer subfolders

**Error: No module named 'timm'**
- Solution: Run `pip install timm`

**Low accuracy**
- Solution: Increase epochs, check dataset quality, verify class balance

## Next Steps

After training:
1. The trained model will be saved in `trained_models/`
2. Use this model in the Django backend for inference
3. Integrate with the React frontend for complete AOI system

## Technology Stack

- **PyTorch** - Deep learning framework
- **torchvision** - Image transformations
- **timm** - Pretrained models library
- **tqdm** - Progress bars
- **Pillow** - Image processing
