# Transfer Learning Project

A comprehensive transfer learning study comparing multiple vision model architectures (VGG19, YOLOv5, ResNet50, Vision Transformer-B) fine-tuned on the Flowers-102 classification task. This project evaluates how different pre-trained architectures perform when adapted for fine-grained visual classification with limited task-specific data.

## Overview

Transfer learning leverages pre-trained models from large-scale datasets (ImageNet) to solve downstream tasks more effectively. This project investigates how different model architectures compare when fine-tuned on the Flowers-102 dataset, including:

- **VGG19**: CNN-based vision model
- **YOLOv5**: Object detection model, adjusted for classification
- **ResNet50**: A classical CNN-based architecture
- **Vision Transformer-B (ViT-B-16)**: A transformer-based architecture with base model size

Each model is trained twice with different random seeds to ensure reproducibility and robustness of results.

## Dataset

### Flowers-102

- **Classes**: 102 different flower species
- **Total Images**: 8,189 images across all splits
- **Image Format**: RGB JPG images
- **Data Split**: 50% training, 25% validation, 25% test
- **Preprocessing**: Images resized to 224×224 pixels with center crop and normalization
- **Augmentation**: Horizontal flips applied during training

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
pip install -r requirements.txt
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
- Consistent results across multiple runs validate model reliability

## Dependencies

- **PyTorch**
- **torchvision**
- **scipy**
- **matplotlib**

## References

- He et al., 2016: "Deep Residual Learning for Image Recognition" (ResNet)
- Dosovitskiy et al., 2020: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Vision Transformer)
- Nilsback & Zisserman, 2008: "Automated Flower Classification over a Large Number of Classes" (Flowers-102 Dataset)
