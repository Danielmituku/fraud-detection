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
| **Random Forest** ⭐ | **0.7047** | **0.6496** | 0.6537 | **0.6456** | **0.8340** |
| Logistic Regression | 0.6468 | 0.6492 | **0.7184** | 0.5922 | 0.8277 |

⭐ **Best Model**: Random Forest (selected based on highest AUC-PR and balanced recall)

### Key Findings
- **Best Performer**: Random Forest achieves the best AUC-PR (0.7047) and recall (64.56%)
- **Precision vs Recall Trade-off**: Logistic Regression has higher precision but lower recall
- **Fraud Detection Rate**: ~65% of fraudulent transactions correctly identified
- **False Positive Rate**: Reasonable precision ensures minimal customer friction

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

