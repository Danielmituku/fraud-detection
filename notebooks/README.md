# Notebooks

This directory contains Jupyter notebooks for the fraud detection analysis.

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `eda-fraud-data.ipynb` | Exploratory Data Analysis for e-commerce fraud data |
| `eda-creditcard.ipynb` | Exploratory Data Analysis for credit card fraud data |
| `feature-engineering.ipynb` | Feature engineering and data transformation |
| `modeling.ipynb` | Model building, training, and evaluation |
| `shap-explainability.ipynb` | SHAP analysis and model interpretation |

## Running the Notebooks

1. Ensure you have installed all dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Run notebooks in order for best results.

## Data Requirements

The notebooks expect the following datasets in `../data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

