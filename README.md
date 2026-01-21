# Transfer Learning Project

A comprehensive transfer learning study comparing multiple vision model architectures (ResNet50, Vision Transformer-B, and Vision Transformer-Huge) fine-tuned on the Flowers-102 classification task. This project evaluates how different pre-trained architectures perform when adapted for fine-grained visual classification with limited task-specific data.

## Overview

Transfer learning leverages pre-trained models from large-scale datasets (ImageNet) to solve downstream tasks more effectively. This project investigates how different model architectures compare when fine-tuned on the Flowers-102 dataset, including:

- **ResNet50**: A classical CNN-based architecture
- **Vision Transformer-B (ViT-B-16)**: A transformer-based architecture with base model size
- **Vision Transformer-Huge (ViT-Huge)**: A larger transformer-based architecture

Each model is trained twice with different random seeds to ensure reproducibility and robustness of results.

## Features

- ✅ **Multiple Architectures**: Compare CNN and transformer-based models
- ✅ **Data Augmentation**: Horizontal flips and center crops for improved generalization
- ✅ **Early Stopping**: Prevents overfitting in transformer models
- ✅ **Reproducible Results**: Multiple runs with different seeds
- ✅ **Comprehensive Evaluation**: Training/validation/test splits and per-epoch metrics
- ✅ **Pretrained Weights**: All models initialized with ImageNet-pretrained weights

## Dataset

### Flowers-102

- **Classes**: 102 different flower species
- **Total Images**: 8,189 images across all splits
- **Image Format**: RGB JPG images
- **Data Split**: 50% training, 25% validation, 25% test
- **Preprocessing**: Images resized to 224×224 pixels with center crop and normalization
- **Augmentation**: Horizontal flips applied during training

**Location**: `data/flowers-102/`
- `jpg/`: Image files organized by class
- `imagelabels.mat`: Class labels for each image
- `setid.mat`: Train/validation/test split indices

## Models & Architecture

### ResNet50
- **Backbone**: ResNet50 pretrained on ImageNet
- **Training Strategy**: Fine-tune all layers
- **Hyperparameters**:
  - Epochs: 5
  - Batch Size: 32
  - Learning Rate: 0.0001
  - Optimizer: Adam
  - Output Classes: 102
- **Runs**: 2 (with seeds 42 and 43)
- **Files**: `models/resnet50_run_1.pth`, `models/resnet50_run_2.pth`

### Vision Transformer-B-16 (ViT-B-16)
- **Backbone**: Vision Transformer-B-16 pretrained on ImageNet
- **Training Strategy**: Fine-tune all layers with early stopping
- **Hyperparameters**:
  - Max Epochs: 20 (with early stopping, patience=3)
  - Batch Size: 16
  - Learning Rate: 0.00001
  - Optimizer: Adam
  - Output Classes: 102
- **Runs**: 2 (with seeds 42 and 43)
- **Files**: `models/vit_b_run_1.pth`, `models/vit_b_run_2.pth`
- **Performance**: ~97.7% validation accuracy

### Vision Transformer-Huge (ViT-Huge)
- **Backbone**: Vision Transformer-Huge pretrained on ImageNet
- **Training Strategy**: Fine-tune all layers with early stopping
- **Hyperparameters**:
  - Max Epochs: 20 (with early stopping, patience=3)
  - Batch Size: 16 (reduced for memory efficiency)
  - Learning Rate: 0.00001
  - Optimizer: Adam
  - Output Classes: 102
- **Runs**: 2 (with seeds 42 and 43)
- **Files**: `models/vithuge_run_1.pth`, `models/vithuge_run_2.pth`

## Project Structure

```
transfer-learning/
├── README.md                    # This file
├── data/
│   └── flowers-102/            # Flowers-102 dataset
│       ├── jpg/                # Image files by class
│       ├── imagelabels.mat     # Class labels
│       └── setid.mat           # Train/val/test indices
├── models/                      # Trained model checkpoints
│   ├── resnet50_run_1.pth
│   ├── resnet50_run_2.pth
│   ├── vit_b_run_1.pth
│   ├── vit_b_run_2.pth
│   ├── vithuge_run_1.pth
│   └── vithuge_run_2.pth
└── src/
    ├── eval/                   # Evaluation scripts
    │   ├── resnet50.py         # ResNet50 training script
    │   ├── vit_b.py            # ViT-B-16 training script
    │   └── utils.py            # Utility functions (EarlyStopping, TransformWrapper)
    └── plots/
        └── plot_results.py     # Visualization script for results
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended for GPU training)
- Required packages: torchvision, scipy, matplotlib

### Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision scipy matplotlib
```

## Usage

### Running Evaluation Scripts

Each evaluation script trains a specific model architecture on the Flowers-102 dataset.

#### ResNet50 Training
```bash
cd src/eval
python resnet50.py
```

#### Vision Transformer-B Training
```bash
cd src/eval
python vit_b.py
```

#### Visualizing Results
```bash
cd src/plots
python plot_results.py
```

### Output

Training scripts generate:
- **Trained Model**: Checkpoint file saved to `models/` directory
- **Console Logs**: Per-epoch training loss, accuracy, validation loss, and validation accuracy
- **Metrics**: Final test set performance

## Reproducibility

All training runs use fixed random seeds to ensure reproducible results:
- **Seed 1**: 42
- **Seed 2**: 43

This allows for:
- Comparison of model consistency across different random initializations
- Verification of reported results
- Fair comparison between architectures

The same data splits (train/val/test) are used across all models to ensure fair evaluation.

## Key Findings

- Vision Transformers demonstrate strong performance on the Flowers-102 classification task
- ViT-B-16 achieves approximately 97.7% validation accuracy
- Transformer models benefit from early stopping to prevent overfitting
- Consistent results across multiple runs validate model reliability

## Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision models and utilities
- **scipy**: Scientific computing (MAT file loading)
- **matplotlib**: Visualization and plotting

## References

- He et al., 2016: "Deep Residual Learning for Image Recognition" (ResNet)
- Dosovitskiy et al., 2020: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Vision Transformer)
- Nilsback & Zisserman, 2008: "Automated Flower Classification over a Large Number of Classes" (Flowers-102 Dataset)
