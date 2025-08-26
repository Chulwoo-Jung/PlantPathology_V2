# 🌱 Plant Pathology Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning solution for plant pathology classification using Vision Transformers (ViT) to identify plant diseases from leaf images.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a deep learning solution for plant pathology classification, designed to identify four different plant conditions:

- **Healthy** - Healthy plant leaves
- **Multiple Diseases** - Leaves with multiple disease symptoms  
- **Rust** - Leaves affected by rust disease
- **Scab** - Leaves affected by scab disease

The solution uses a **DeiT3 (Data-efficient image Transformers)** model architecture with a two-phase fine-tuning strategy for optimal performance.

## ✨ Features

- 🚀 **State-of-the-art Architecture**: DeiT3 Vision Transformer with 83% ImageNet-1k accuracy
- 🔄 **Two-Phase Fine-tuning**: Efficient training strategy for better convergence
- 🎨 **Advanced Augmentation**: Comprehensive image augmentation pipeline using Albumentations
- 📊 **Smart Data Loading**: Automatically handles datasets with or without labels
- 🎯 **Advanced Training**: Early stopping, learning rate scheduling, and AUROC tracking
- 🔧 **Flexible Design**: Easy to customize and extend for different use cases

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-pathology.git
cd plant-pathology

# Create virtual environment
python -m venv plant_venv
source plant_venv/bin/activate  # On Windows: plant_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```bash
pip install torch torchvision
pip install timm albumentations
pip install torchmetrics tqdm
pip install opencv-python matplotlib pandas
pip install jupyter scikit-learn
```

## 📖 Usage

### 1. Data Preparation

```python
import pandas as pd
from plant_pathology_data import plant_data_loader

# Load training data
train_df = pd.read_csv('train.csv')
train_df['image_path'] = 'images/' + train_df['image_id'] + '.jpg'

# Split into train/validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['target'], 
    random_state=42
)
```

### 2. Data Loading

```python
# Create data loaders
train_loader, val_loader = plant_data_loader(
    df=train_df, 
    val_df=val_df, 
    batch_size=16,
    train_transform=train_transform,
    val_transform=val_transform
)

# For test data (no labels)
test_loader = plant_data_loader(
    df=test_df, 
    batch_size=16,
    train_transform=test_transform,
    is_test=True
)
```

### 3. Model Training

```python
from trainer_V2 import Trainer
from timm import create_model

# Create model
model = create_model(
    'deit3_small_patch16_224.fb_in22k_ft_in1k', 
    pretrained=True, 
    num_classes=4
)

# Phase 1: Train only classification head
model = freeze_feature_extractor(model)
trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler)
history = trainer.fit(epochs=30)

# Phase 2: Fine-tune all layers
for param in model.parameters():
    param.requires_grad = True
trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler)
history = trainer.fit(epochs=30)
```

### 4. Prediction

```python
from predictor_V2 import Predictor

# Load trained model
model.load_state_dict(torch.load('state_dict'))

# Make predictions
predictor = Predictor(model)
probabilities = predictor.predict_proba(test_images)
predictions = predictor.predict(test_images)
```

## 📁 Project Structure

```
plant_pathology/
├── 📊 Data & Models
│   ├── images/                    # Training and test images
│   ├── train.csv                  # Training data with labels
│   ├── test.csv                   # Test data for predictions
│   ├── sample_submission.csv      # Submission format example
│   └── state_dict                 # Trained model weights
│
├── 🐍 Core Modules
│   ├── plant_pathology_data.py    # Data loading and preprocessing
│   ├── trainer_V2.py             # Training loop and validation
│   └── predictor_V2.py           # Prediction and inference
│
├── 📓 Jupyter Notebooks
│   ├── 1_check_images.ipynb      # Data exploration and visualization
│   └── 2_training_model.ipynb    # Model training and evaluation
│
└── 📚 Additional Files
    ├── train_df.pkl               # Preprocessed training data
    ├── test_df.pkl                # Preprocessed test data
    └── .gitignore                 # Git ignore patterns
```

## 🏗️ Model Architecture

- **Base Model**: DeiT3 Small Patch16 224
- **Input Size**: 224×224×3 RGB images
- **Output**: 4-class classification with probability scores
- **Pretrained**: ImageNet-22k → ImageNet-1k fine-tuned
- **Performance**: 83% accuracy on ImageNet-1k

## 🎯 Training Strategy

### Two-Phase Fine-tuning

1. **Phase 1: Head Training**
   - Freeze feature extractor backbone
   - Train only classification head
   - Quick convergence with frozen features

2. **Phase 2: Full Fine-tuning**
   - Unfreeze all layers
   - Fine-tune entire model
   - Further performance improvement

### Training Features

- **Optimizer**: Adam with configurable learning rate
- **Scheduler**: ReduceLROnPlateau with patience
- **Loss Function**: CrossEntropyLoss
- **Metrics**: AUROC (Area Under ROC Curve)
- **Early Stopping**: Configurable patience
- **Data Augmentation**: Comprehensive pipeline

## 🎨 Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.VerticalFlip(p=0.4),
    A.HorizontalFlip(p=0.4),
    A.Rotate(limit=15, p=0.4),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=8, contrast_limit=8, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ], p=0.3),
    A.Blur(blur_limit=3, p=0.3),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

## 📈 Results

The model achieves competitive performance through the two-phase fine-tuning approach:

- **Phase 1 (Head-only)**: Quick convergence with frozen backbone
- **Phase 2 (Full fine-tuning)**: Further improvement by fine-tuning all layers
- **Final Performance**: High AUROC scores on validation set

Training curves show consistent improvement in both loss and AUROC metrics across epochs.

## 🔧 Customization

### Model Architecture

```python
# Change to different Vision Transformer
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=4)

# Or use CNN-based models
model = create_model('resnet50', pretrained=True, num_classes=4)
model = create_model('efficientnet_b0', pretrained=True, num_classes=4)
```

### Training Parameters

```python
# Customize training parameters
trainer = Trainer(
    model, train_loader, val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    early_patience=15,        # Early stopping patience
    early_stop=True,          # Enable/disable early stopping
    cutmix=True,              # Enable CutMix augmentation
    cutmix_prob=0.5          # CutMix probability
)
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/plant-pathology/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/plant-pathology/discussions)

## 🙏 Acknowledgments

- [DeiT3 Paper](https://arxiv.org/abs/2204.07118) for the Vision Transformer architecture
- [Albumentations](https://albumentations.ai/) for image augmentation
- [timm](https://github.com/huggingface/pytorch-image-models) for model implementations

---

**Note**: This project is designed for educational and research purposes. The trained models and datasets should be used in accordance with their respective licenses and terms of use.

