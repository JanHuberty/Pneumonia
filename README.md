![CI](https://github.com/JanHuberty/Pneumonia/actions/workflows/python-ci.yml/badge.svg)
![Links](https://github.com/JanHuberty/Pneumonia/actions/workflows/links.yml/badge.svg)
![Auto format](https://github.com/JanHuberty/Pneumonia/actions/workflows/auto-format.yml/badge.svg)
![License](https://img.shields.io/github/license/JanHuberty/Pneumonia)


# Pneumonia Detection from Chest X-rays (CNN)

ResNet-based classifier trained on chest X-rays.  
**Stack:** PyTorch, torchvision, Python 3.x

## Overview
- Transfer learning (frozen → fine-tune)
- Metrics: accuracy, ROC/AUC, confusion matrix
- Reproducible seeds, checkpoints saved

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Option A: run notebook in notebooks/
# Option B: run the script
# python src/train.py --epochs 1 --seed 42
