![CI](https://github.com/JanHuberty/Pneumonia/actions/workflows/python-ci.yml/badge.svg)
![Links](https://github.com/JanHuberty/Pneumonia/actions/workflows/links.yml/badge.svg)
![Auto format](https://github.com/JanHuberty/Pneumonia/actions/workflows/auto-format.yml/badge.svg)
![License](https://img.shields.io/github/license/JanHuberty/Pneumonia)


# Pneumonia Detection from Chest X-rays (CNN)

**(Optional but great)** Add an **Open in Colab** badge if you have a notebook:

```md
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/JanHuberty/Pneumonia/blob/main/notebooks/train.ipynb)

ResNet-based classifier trained on chest X-rays.  
**Stack:** PyTorch, torchvision, Python 3.x

## Overview
- Transfer learning (frozen → fine-tune)
- Metrics: accuracy, ROC/AUC, confusion matrix
- Reproducible seeds, checkpoints saved

## Quick Start
```bash
# 1) Create & activate env
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 2) Install deps (CPU-safe)
pip install -r requirements.txt

# 3) Option A: Run the notebook
# Open notebooks/train.ipynb and run all cells.

# 3) Option B: Run the training script (if present)
# python src/train.py --epochs 1 --seed 42 --data_dir ./data/sample

# 4) Evaluate (example)
# python src/eval.py --checkpoint checkpoints/best.pth --data_dir ./data/sample

