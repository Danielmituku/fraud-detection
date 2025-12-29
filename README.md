# Fraud Detection System

A comprehensive fraud detection solution for e-commerce and bank credit transactions using machine learning and explainable AI.

## 🎯 Project Overview

This project implements fraud detection models to identify fraudulent transactions in:
- **E-commerce transactions** (Fraud_Data.csv)
- **Bank credit transactions** (creditcard.csv)

The system leverages geolocation analysis, transaction pattern recognition, and advanced ML techniques to accurately detect fraud while minimizing false positives.

## 📁 Project Structure

```
fraud-detection/
├── .vscode/                    # VS Code settings
├── .github/workflows/          # CI/CD workflows
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb    # EDA for e-commerce data
│   ├── eda-creditcard.ipynb    # EDA for credit card data
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/                        # Source code modules
├── tests/                      # Unit tests
├── models/                     # Saved model artifacts
├── scripts/                    # Utility scripts
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fraud-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the datasets and place them in `data/raw/`:
   - [Fraud E-commerce Dataset](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)
   - [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 📊 Datasets

### Fraud_Data.csv (E-commerce)
| Feature | Description |
|---------|-------------|
| user_id | Unique user identifier |
| signup_time | User registration timestamp |
| purchase_time | Transaction timestamp |
| purchase_value | Transaction amount ($) |
| device_id | Device identifier |
| source | Traffic source (SEO, Ads, etc.) |
| browser | Browser used |
| sex | User gender |
| age | User age |
| ip_address | IP address |
| class | Target (1=fraud, 0=legitimate) |

### IpAddress_to_Country.csv
| Feature | Description |
|---------|-------------|
| lower_bound_ip_address | IP range lower bound |
| upper_bound_ip_address | IP range upper bound |
| country | Country name |

### creditcard.csv (Bank Transactions)
| Feature | Description |
|---------|-------------|
| Time | Seconds since first transaction |
| V1-V28 | PCA-transformed features |
| Amount | Transaction amount ($) |
| Class | Target (1=fraud, 0=legitimate) |

## 🔬 Methodology

### Task 1: Data Analysis & Preprocessing
- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Geolocation integration (IP → Country mapping)
- Feature engineering (time-based, velocity features)
- Class imbalance handling (SMOTE)

### Task 2: Model Building
- Baseline: Logistic Regression
- Ensemble: Random Forest, XGBoost, LightGBM
- Stratified K-Fold cross-validation
- Metrics: AUC-PR, F1-Score, Precision, Recall

### Task 3: Model Explainability
- SHAP Summary Plots
- Force Plots for individual predictions
- Business recommendations

## 📈 Key Features

- **Geolocation Analysis**: Map IP addresses to countries for fraud pattern detection
- **Velocity Features**: Track transaction frequency per user
- **Time-based Features**: Hour of day, day of week, time since signup
- **SHAP Explainability**: Understand model decisions

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src
```

## 📋 Results Summary

### E-commerce Fraud Detection Models

| Model | AUC-PR | F1-Score | Precision | Recall | ROC-AUC |
|-------|--------|----------|-----------|--------|---------|
| **Random Forest (Tuned)** ⭐ | **0.7126** | 0.6277 | 0.5815 | **0.6820** | 0.8423 |
| Gradient Boosting | 0.7117 | **0.7015** | **0.9987** | 0.5406 | **0.8434** |
| Logistic Regression | 0.6643 | 0.6043 | 0.5326 | 0.6982 | 0.8411 |

⭐ **Best E-commerce Model**: Random Forest (Tuned) - Highest AUC-PR with balanced recall

### Credit Card Fraud Detection Models

| Model | AUC-PR | F1-Score | Precision | Recall | ROC-AUC |
|-------|--------|----------|-----------|--------|---------|
| **Gradient Boosting** ⭐ | **0.8583** | **0.7685** | **0.7034** | 0.8469 | 0.9766 |
| Random Forest (Tuned) | 0.7773 | 0.4735 | 0.3257 | 0.8673 | **0.9798** |
| Logistic Regression | 0.7256 | 0.1092 | 0.0580 | **0.9184** | 0.9688 |

⭐ **Best Credit Card Model**: Gradient Boosting - Best balance of precision and recall

### Cross-Validation Results (5-Fold)

| Dataset | Model | CV F1 (mean ± std) | CV ROC-AUC (mean ± std) |
|---------|-------|-------------------|------------------------|
| E-commerce | Gradient Boosting | 0.9254 ± 0.0023 | 0.9672 ± 0.0015 |
| E-commerce | Random Forest | 0.8323 ± 0.0027 | 0.9539 ± 0.0016 |
| Credit Card | Random Forest | 0.7139 ± 0.1071 | 0.9294 ± 0.0200 |

### Key Findings
- **E-commerce**: Random Forest achieves best AUC-PR (0.7126) with 68.2% recall
- **Credit Card**: Gradient Boosting achieves best AUC-PR (0.8583) with balanced metrics
- **Feature Scaling**: StandardScaler applied to all numerical features
- **Hyperparameter Tuning**: GridSearchCV used for optimal model configuration

## 👥 Team

- **Tutors**: Kerod, Mahbubah, Filimon

## 📅 Timeline

- Interim-1: December 21, 2025
- Interim-2: December 28, 2025
- Final Submission: December 30, 2025

## 📚 References

- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Kaggle: Fraud E-commerce Dataset](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 📄 License

This project is for educational purposes as part of 10Academy Week 5 Challenge.

