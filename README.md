# Provably Label-noise Learning (PLM)

This repository contains implementation of provably label-noise learning methods combining techniques from multiple research papers.

## Overview

This project implements various approaches for learning with noisy labels, including:

- **Label Noise Learning (LNL)**: Core algorithms for handling noisy labels in datasets
- **Contrastive Learning**: Methods for learning robust representations
- **K-means Refinement**: Clustering-based label refinement techniques
- **Human-in-the-loop Learning**: Interactive approaches for label correction

## Project Structure

```
├── data/                           # Dataset files
│   ├── CIFAR-10_human.pt          # CIFAR-10 with human annotations
│   ├── CIFAR-10_human_ordered.npy
│   ├── CIFAR-100_human.pt         # CIFAR-100 with human annotations
│   └── CIFAR-100_human_ordered.npy
├── models/                         # Model architectures
│   ├── models.py                   # Main model definitions
│   └── resnet.py                   # ResNet implementation
├── utils/                          # Utility functions
│   ├── attention_crop.py           # Attention-based cropping
│   ├── combined_dataset.py         # Dataset combination utilities
│   ├── custom_dataset.py           # Custom dataset implementations
│   ├── utils_algo.py               # Algorithm utilities
│   └── utils_data.py               # Data processing utilities
├── res/                            # Results and trained models
│   ├── cifar-10/                   # CIFAR-10 experimental results
│   └── cifar-100/                  # CIFAR-100 experimental results
├── lnl.py                         # Main label noise learning implementation
├── lnl-human.py                   # Human-in-the-loop variant
├── lnl_combined.py                # Combined approach
├── labeling*.py                   # Various labeling strategies
├── compare_approaches.py          # Method comparison scripts
└── test_*.py                      # Testing and evaluation scripts
```

## Features

### Core Algorithms
- **Provably End-to-End Label Noise Learning**: Implementation without anchor points
- **T-Revision**: Transition matrix revision techniques
- **Unsupervised Classification**: Clustering-based approaches

### Datasets Supported
- CIFAR-10
- CIFAR-100
- Custom datasets with human annotations

### Training Methods
- Standard supervised learning
- Contrastive learning with noisy labels
- Semi-supervised learning with label refinement
- Human-in-the-loop correction

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd ALNL
```

2. Install required dependencies:
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## Usage

### Basic Label Noise Learning
```bash
python lnl.py --dataset cifar10 --noise_rate 0.2 --epochs 40
```

### Human-in-the-Loop Learning
```bash
python lnl-human.py --dataset cifar10 --noise_rate 0.2 --epochs 40
```

### Combined Approach
```bash
python lnl_combined.py --dataset cifar10 --noise_rate 0.2 --epochs 40
```

