# Pneumonia Detection from Chest X-rays (CNN)

ResNet-based classifier trained on chest X-rays.  
**Stack:** PyTorch, torchvision, Python 3.x

## Overview
- Transfer learning (frozen → fine-tune)
- Metrics: accuracy, ROC/AUC, confusion matrix
- Reproducible seeds, checkpoints saved

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
