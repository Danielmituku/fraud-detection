# Changelog

All notable changes to the Fraud Detection project will be documented in this file.

## [1.0.0] - 2025-12-24

### Added
- **Task 1: Data Analysis and Preprocessing**
  - Exploratory Data Analysis notebooks for both e-commerce and credit card datasets
  - Data cleaning and loading utilities (`src/data_loader.py`)
  - IP to country geolocation mapping using range-based lookup
  - Feature engineering module with time-based, velocity, and device features
  - SMOTE implementation for handling class imbalance
  - Comprehensive visualizations for EDA

- **Task 2: Model Building and Training**
  - Baseline Logistic Regression model
  - Ensemble models: Random Forest, XGBoost, LightGBM
  - Stratified train-test split preserving class distribution
  - Model evaluation with appropriate metrics (F1-Score, AUC-PR, ROC-AUC)
  - Model comparison and selection with justification
  - Cross-validation support

- **Task 3: Model Explainability**
  - SHAP (SHapley Additive exPlanations) integration
  - Summary plots for global feature importance
  - Force plots for individual prediction explanation
  - Business recommendations based on model insights

- **Project Infrastructure**
  - GitHub Actions CI/CD workflow for automated testing
  - Unit tests for data loader and feature engineering modules
  - Comprehensive project documentation (README.md)
  - Requirements file with pinned dependencies
  - VS Code settings for development

### Key Insights
- Time since signup is a critical fraud predictor
- Devices shared by multiple users indicate potential fraud rings
- Extreme class imbalance (~9:1 to ~500:1) requires careful handling
- Geographic patterns contribute to fraud detection

## [1.1.0] - 2025-12-29 (Interim-2)

### Added
- **Task 2: Model Building and Training**
  - Trained Logistic Regression baseline model
  - Trained Random Forest ensemble model
  - Stratified train-test split with 80/20 ratio
  - SMOTE implementation (0.5 sampling strategy)
  - Model evaluation with AUC-PR, F1-Score, ROC-AUC
  - Model comparison and selection pipeline
  - Saved model artifacts to `models/` directory

- **Task 3: Model Explainability**
  - SHAP TreeExplainer integration
  - Global feature importance (Summary Plot, Bar Plot)
  - Individual prediction explanations (Force Plots)
  - Business recommendations based on SHAP insights
  - Saved SHAP artifacts and recommendations

- **Training Scripts**
  - `scripts/train_models.py` - Automated model training
  - `scripts/run_shap_analysis.py` - SHAP explainability pipeline

### Results
- **Best Model**: Random Forest
- **AUC-PR**: 0.7047
- **F1-Score**: 0.6496
- **Recall**: 64.56%

### Key Findings
- Weekend transactions show highest impact on fraud prediction
- Time since signup remains critical fraud indicator
- Device sharing patterns (device_unique_users) important for fraud rings
- Temporal features dominate over demographic features

## [Unreleased]
- API deployment with Flask/FastAPI
- Real-time scoring pipeline
- Dashboard for monitoring fraud patterns
- Additional feature engineering based on production data

