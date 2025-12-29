"""
Final Model Training Script for Fraud Detection
Includes: Cross-validation, Feature scaling, Hyperparameter tuning
Trains on both Fraud_Data.csv and creditcard.csv datasets

Author: Daniel Mituku
Date: December 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Import project modules
from src.data_loader import load_fraud_data, load_ip_to_country, map_ip_to_country, load_creditcard_data
from src.feature_engineering import (
    create_time_features, create_transaction_velocity_features,
    create_device_features, encode_categorical_features,
    prepare_features_for_modeling, scale_numerical_features
)
from src.modeling import (
    stratified_train_test_split, apply_smote,
    evaluate_model, compare_models, save_model
)

print("=" * 80)
print("FINAL MODEL TRAINING - FRAUD DETECTION")
print("Cross-Validation | Feature Scaling | Hyperparameter Tuning")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# PART 1: E-COMMERCE FRAUD DATA (Fraud_Data.csv)
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: E-COMMERCE FRAUD DETECTION (Fraud_Data.csv)")
print("=" * 80)

# 1.1 Load and prepare data
print("\n1.1 Loading and preparing e-commerce data...")
fraud_df = load_fraud_data('data/raw/Fraud_Data.csv')
ip_country_df = load_ip_to_country('data/raw/IpAddress_to_Country.csv')

# Apply feature engineering pipeline
fraud_df = map_ip_to_country(fraud_df, ip_country_df)
fraud_df = create_time_features(fraud_df)
fraud_df = create_transaction_velocity_features(fraud_df)
fraud_df = create_device_features(fraud_df)
fraud_df, _ = encode_categorical_features(fraud_df, ['source', 'browser', 'sex', 'country'])

# Prepare features
X_ecom, y_ecom = prepare_features_for_modeling(fraud_df, target_col='class')
print(f"   Features shape: {X_ecom.shape}")
print(f"   Class distribution: {dict(y_ecom.value_counts())}")

# 1.2 Feature Scaling
print("\n1.2 Applying StandardScaler...")
scaler_ecom = StandardScaler()
X_ecom_scaled = pd.DataFrame(
    scaler_ecom.fit_transform(X_ecom),
    columns=X_ecom.columns,
    index=X_ecom.index
)
print("   ✓ Features scaled")

# 1.3 Train-test split
print("\n1.3 Stratified train-test split...")
X_train_ecom, X_test_ecom, y_train_ecom, y_test_ecom = train_test_split(
    X_ecom_scaled, y_ecom, test_size=0.2, random_state=42, stratify=y_ecom
)
print(f"   Training: {X_train_ecom.shape[0]} samples")
print(f"   Test: {X_test_ecom.shape[0]} samples")

# 1.4 Apply SMOTE
print("\n1.4 Applying SMOTE to training data...")
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_ecom_smote, y_train_ecom_smote = smote.fit_resample(X_train_ecom, y_train_ecom)
print(f"   After SMOTE: {pd.Series(y_train_ecom_smote).value_counts().to_dict()}")

# 1.5 Cross-Validation with Stratified K-Fold
print("\n1.5 Cross-Validation (Stratified 5-Fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models for CV
cv_models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
}

cv_results_ecom = {}
for name, model in cv_models.items():
    print(f"   Evaluating {name}...")
    
    # Multiple scoring metrics
    f1_scores = cross_val_score(model, X_train_ecom_smote, y_train_ecom_smote, cv=cv, scoring='f1')
    roc_scores = cross_val_score(model, X_train_ecom_smote, y_train_ecom_smote, cv=cv, scoring='roc_auc')
    precision_scores = cross_val_score(model, X_train_ecom_smote, y_train_ecom_smote, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X_train_ecom_smote, y_train_ecom_smote, cv=cv, scoring='recall')
    
    cv_results_ecom[name] = {
        'F1-Score': {'mean': f1_scores.mean(), 'std': f1_scores.std()},
        'ROC-AUC': {'mean': roc_scores.mean(), 'std': roc_scores.std()},
        'Precision': {'mean': precision_scores.mean(), 'std': precision_scores.std()},
        'Recall': {'mean': recall_scores.mean(), 'std': recall_scores.std()}
    }
    print(f"      F1: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
    print(f"      ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")

# 1.6 Hyperparameter Tuning with GridSearchCV
print("\n1.6 Hyperparameter Tuning (GridSearchCV)...")

# Random Forest hyperparameter grid (reduced for speed)
rf_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [8, 12],
    'class_weight': ['balanced']
}

print("   Tuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3,  # Use 3-fold for speed
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_ecom_smote, y_train_ecom_smote)

print(f"   Best params: {rf_grid.best_params_}")
print(f"   Best CV F1: {rf_grid.best_score_:.4f}")

best_rf_ecom = rf_grid.best_estimator_

# 1.7 Final Model Evaluation on Test Set
print("\n1.7 Final Evaluation on Test Set...")

# Train final models
lr_ecom = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_ecom.fit(X_train_ecom_smote, y_train_ecom_smote)

gb_ecom = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb_ecom.fit(X_train_ecom_smote, y_train_ecom_smote)

# Evaluate all models
models_ecom = {
    'Logistic Regression': lr_ecom,
    'Random Forest (Tuned)': best_rf_ecom,
    'Gradient Boosting': gb_ecom
}

results_ecom = {}
for name, model in models_ecom.items():
    y_pred = model.predict(X_test_ecom)
    y_proba = model.predict_proba(X_test_ecom)[:, 1]
    
    results_ecom[name] = {
        'Precision': precision_score(y_test_ecom, y_pred),
        'Recall': recall_score(y_test_ecom, y_pred),
        'F1-Score': f1_score(y_test_ecom, y_pred),
        'ROC-AUC': roc_auc_score(y_test_ecom, y_proba),
        'AUC-PR': average_precision_score(y_test_ecom, y_proba)
    }

# Display results
print("\n" + "=" * 80)
print("E-COMMERCE FRAUD DETECTION - FINAL RESULTS")
print("=" * 80)
results_df_ecom = pd.DataFrame(results_ecom).T.round(4)
print(results_df_ecom.to_string())

# Best model selection
best_model_name_ecom = max(results_ecom, key=lambda x: results_ecom[x]['AUC-PR'])
best_model_ecom = models_ecom[best_model_name_ecom]
print(f"\n✓ Best E-commerce Model: {best_model_name_ecom} (AUC-PR: {results_ecom[best_model_name_ecom]['AUC-PR']:.4f})")

# ============================================================================
# PART 2: CREDIT CARD FRAUD DATA (creditcard.csv)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: CREDIT CARD FRAUD DETECTION (creditcard.csv)")
print("=" * 80)

# 2.1 Load data
print("\n2.1 Loading credit card data...")
cc_df = load_creditcard_data('data/raw/creditcard.csv')
print(f"   Shape: {cc_df.shape}")
print(f"   Class distribution: {dict(cc_df['Class'].value_counts())}")

# 2.2 Prepare features (already PCA transformed, just need Amount and Time scaling)
print("\n2.2 Preparing features...")
X_cc = cc_df.drop(columns=['Class'])
y_cc = cc_df['Class']

# Scale Amount and Time (V features are already scaled via PCA)
scaler_cc = StandardScaler()
X_cc[['Time', 'Amount']] = scaler_cc.fit_transform(X_cc[['Time', 'Amount']])
print("   ✓ Time and Amount scaled")

# 2.3 Train-test split
print("\n2.3 Stratified train-test split...")
X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(
    X_cc, y_cc, test_size=0.2, random_state=42, stratify=y_cc
)
print(f"   Training: {X_train_cc.shape[0]} samples")
print(f"   Test: {X_test_cc.shape[0]} samples")

# 2.4 Apply SMOTE (carefully due to extreme imbalance)
print("\n2.4 Applying SMOTE to training data...")
smote_cc = SMOTE(random_state=42, sampling_strategy=0.1)  # Lower ratio due to extreme imbalance
X_train_cc_smote, y_train_cc_smote = smote_cc.fit_resample(X_train_cc, y_train_cc)
print(f"   After SMOTE: {pd.Series(y_train_cc_smote).value_counts().to_dict()}")

# 2.5 Cross-Validation
print("\n2.5 Cross-Validation (Stratified 5-Fold)...")

# Use sample for CV speed
cv_sample_size = min(30000, len(X_train_cc_smote))
X_cv_sample = X_train_cc_smote.iloc[:cv_sample_size]
y_cv_sample = y_train_cc_smote.iloc[:cv_sample_size]

cv_results_cc = {}
for name, model in cv_models.items():
    print(f"   Evaluating {name}...")
    
    f1_scores = cross_val_score(model, X_cv_sample, y_cv_sample, cv=cv, scoring='f1')
    roc_scores = cross_val_score(model, X_cv_sample, y_cv_sample, cv=cv, scoring='roc_auc')
    
    cv_results_cc[name] = {
        'F1-Score': {'mean': f1_scores.mean(), 'std': f1_scores.std()},
        'ROC-AUC': {'mean': roc_scores.mean(), 'std': roc_scores.std()}
    }
    print(f"      F1: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
    print(f"      ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")

# 2.6 Hyperparameter Tuning
print("\n2.6 Hyperparameter Tuning (GridSearchCV)...")

# Use a sample for faster tuning
sample_size = min(50000, len(X_train_cc_smote))
X_sample_cc = X_train_cc_smote.iloc[:sample_size]
y_sample_cc = y_train_cc_smote.iloc[:sample_size]

rf_grid_cc = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
rf_grid_cc.fit(X_sample_cc, y_sample_cc)

print(f"   Best params: {rf_grid_cc.best_params_}")
print(f"   Best CV F1: {rf_grid_cc.best_score_:.4f}")

# Retrain on full data with best params
best_rf_cc = RandomForestClassifier(**rf_grid_cc.best_params_, random_state=42, n_jobs=-1)
best_rf_cc.fit(X_train_cc_smote, y_train_cc_smote)

# 2.7 Final Model Evaluation
print("\n2.7 Final Evaluation on Test Set...")

# Train final models
lr_cc = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_cc.fit(X_train_cc_smote, y_train_cc_smote)

gb_cc = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb_cc.fit(X_train_cc_smote, y_train_cc_smote)

models_cc = {
    'Logistic Regression': lr_cc,
    'Random Forest (Tuned)': best_rf_cc,
    'Gradient Boosting': gb_cc
}

results_cc = {}
for name, model in models_cc.items():
    y_pred = model.predict(X_test_cc)
    y_proba = model.predict_proba(X_test_cc)[:, 1]
    
    results_cc[name] = {
        'Precision': precision_score(y_test_cc, y_pred),
        'Recall': recall_score(y_test_cc, y_pred),
        'F1-Score': f1_score(y_test_cc, y_pred),
        'ROC-AUC': roc_auc_score(y_test_cc, y_proba),
        'AUC-PR': average_precision_score(y_test_cc, y_proba)
    }

# Display results
print("\n" + "=" * 80)
print("CREDIT CARD FRAUD DETECTION - FINAL RESULTS")
print("=" * 80)
results_df_cc = pd.DataFrame(results_cc).T.round(4)
print(results_df_cc.to_string())

# Best model selection
best_model_name_cc = max(results_cc, key=lambda x: results_cc[x]['AUC-PR'])
best_model_cc = models_cc[best_model_name_cc]
print(f"\n✓ Best Credit Card Model: {best_model_name_cc} (AUC-PR: {results_cc[best_model_name_cc]['AUC-PR']:.4f})")

# ============================================================================
# PART 3: SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: SAVING MODELS AND RESULTS")
print("=" * 80)

os.makedirs('models', exist_ok=True)

# Save e-commerce models
save_model(best_model_ecom, 'models/best_ecommerce_model.pkl')
save_model(scaler_ecom, 'models/ecommerce_scaler.pkl')

# Save credit card models
save_model(best_model_cc, 'models/best_creditcard_model.pkl')
save_model(scaler_cc, 'models/creditcard_scaler.pkl')

# Save all results
all_results = {
    'ecommerce': {
        'best_model': best_model_name_ecom,
        'best_params': rf_grid.best_params_ if 'Random Forest' in best_model_name_ecom else {},
        'cv_results': {k: {m: {'mean': round(v['mean'], 4), 'std': round(v['std'], 4)} 
                          for m, v in metrics.items()} 
                      for k, metrics in cv_results_ecom.items()},
        'test_results': {k: {m: round(v, 4) for m, v in metrics.items()} 
                        for k, metrics in results_ecom.items()},
        'feature_names': X_ecom.columns.tolist()
    },
    'creditcard': {
        'best_model': best_model_name_cc,
        'best_params': rf_grid_cc.best_params_ if 'Random Forest' in best_model_name_cc else {},
        'cv_results': {k: {m: {'mean': round(v['mean'], 4), 'std': round(v['std'], 4)} 
                          for m, v in metrics.items()} 
                      for k, metrics in cv_results_cc.items()},
        'test_results': {k: {m: round(v, 4) for m, v in metrics.items()} 
                        for k, metrics in results_cc.items()},
        'feature_names': X_cc.columns.tolist()
    },
    'trained_at': datetime.now().isoformat()
}

with open('models/final_model_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("   ✓ Results saved to models/final_model_results.json")

# ============================================================================
# PART 4: GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: GENERATING VISUALIZATIONS")
print("=" * 80)

# 4.1 Cross-validation comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# E-commerce CV results
cv_df_ecom = pd.DataFrame({
    name: {metric: data['mean'] for metric, data in metrics.items()}
    for name, metrics in cv_results_ecom.items()
}).T
cv_df_ecom[['F1-Score', 'ROC-AUC']].plot(kind='bar', ax=axes[0], rot=15)
axes[0].set_title('E-commerce Fraud: Cross-Validation Results (5-Fold)')
axes[0].set_ylabel('Score')
axes[0].set_ylim([0, 1])
axes[0].legend(loc='lower right')
axes[0].grid(axis='y', alpha=0.3)

# Credit card CV results
cv_df_cc = pd.DataFrame({
    name: {metric: data['mean'] for metric, data in metrics.items()}
    for name, metrics in cv_results_cc.items()
}).T
cv_df_cc[['F1-Score', 'ROC-AUC']].plot(kind='bar', ax=axes[1], rot=15)
axes[1].set_title('Credit Card Fraud: Cross-Validation Results (5-Fold)')
axes[1].set_ylabel('Score')
axes[1].set_ylim([0, 1])
axes[1].legend(loc='lower right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/cross_validation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Cross-validation comparison saved")

# 4.2 Final test results comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# E-commerce test results
results_df_ecom.T.plot(kind='bar', ax=axes[0], rot=0)
axes[0].set_title('E-commerce Fraud: Test Set Performance')
axes[0].set_ylabel('Score')
axes[0].set_ylim([0, 1])
axes[0].legend(loc='lower right', fontsize=8)
axes[0].grid(axis='y', alpha=0.3)

# Credit card test results
results_df_cc.T.plot(kind='bar', ax=axes[1], rot=0)
axes[1].set_title('Credit Card Fraud: Test Set Performance')
axes[1].set_ylabel('Score')
axes[1].set_ylim([0, 1])
axes[1].legend(loc='lower right', fontsize=8)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/final_test_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Final test comparison saved")

# 4.3 Confusion matrices for best models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# E-commerce confusion matrix
y_pred_ecom = best_model_ecom.predict(X_test_ecom)
cm_ecom = confusion_matrix(y_test_ecom, y_pred_ecom)
sns.heatmap(cm_ecom, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'E-commerce: {best_model_name_ecom}')

# Credit card confusion matrix
y_pred_cc = best_model_cc.predict(X_test_cc)
cm_cc = confusion_matrix(y_test_cc, y_pred_cc)
sns.heatmap(cm_cc, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Credit Card: {best_model_name_cc}')

plt.tight_layout()
plt.savefig('figures/final_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Final confusion matrices saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("SUMMARY:")
print("-" * 40)
print(f"E-commerce Best Model: {best_model_name_ecom}")
print(f"   - AUC-PR: {results_ecom[best_model_name_ecom]['AUC-PR']:.4f}")
print(f"   - F1-Score: {results_ecom[best_model_name_ecom]['F1-Score']:.4f}")
print()
print(f"Credit Card Best Model: {best_model_name_cc}")
print(f"   - AUC-PR: {results_cc[best_model_name_cc]['AUC-PR']:.4f}")
print(f"   - F1-Score: {results_cc[best_model_name_cc]['F1-Score']:.4f}")
print()
print("Artifacts saved:")
print("   - models/best_ecommerce_model.pkl")
print("   - models/best_creditcard_model.pkl")
print("   - models/final_model_results.json")
print("   - figures/cross_validation_comparison.png")
print("   - figures/final_test_comparison.png")
print("   - figures/final_confusion_matrices.png")

