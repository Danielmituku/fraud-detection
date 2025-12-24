# Scripts

This directory contains utility scripts for the fraud detection project.

## Available Scripts

| Script | Description |
|--------|-------------|
| `download_data.py` | Download datasets from Kaggle |
| `preprocess_data.py` | Run data preprocessing pipeline |
| `train_model.py` | Train and save fraud detection models |
| `evaluate_model.py` | Evaluate saved models |

## Usage

```bash
# Download data
python scripts/download_data.py

# Preprocess data
python scripts/preprocess_data.py

# Train model
python scripts/train_model.py --model xgboost --dataset fraud

# Evaluate model
python scripts/evaluate_model.py --model models/xgboost_fraud.pkl
```

