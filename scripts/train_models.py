"""
Train and evaluate fraud detection models for Interim Report 2.
Generates model artifacts and results summary.
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

# Import project modules
from src.data_loader import load_fraud_data, load_ip_to_country, map_ip_to_country, load_creditcard_data
from src.feature_engineering import (
    create_time_features, create_transaction_velocity_features,
    create_device_features, encode_categorical_features,
    prepare_features_for_modeling
)
from src.modeling import (
    stratified_train_test_split, apply_smote,
    train_logistic_regression, train_random_forest,
    train_xgboost, train_lightgbm,
    evaluate_model, compare_models, save_model,
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
)

print("=" * 70)
print("FRAUD DETECTION MODEL TRAINING - INTERIM REPORT 2")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("1. Loading and preparing data...")

# Load e-commerce fraud data
fraud_df = load_fraud_data('data/raw/Fraud_Data.csv')
ip_country_df = load_ip_to_country('data/raw/IpAddress_to_Country.csv')

# Apply feature engineering pipeline
print("   Applying feature engineering...")
fraud_df = map_ip_to_country(fraud_df, ip_country_df)
fraud_df = create_time_features(fraud_df)
fraud_df = create_transaction_velocity_features(fraud_df)
fraud_df = create_device_features(fraud_df)
fraud_df, label_encoders = encode_categorical_features(fraud_df, ['source', 'browser', 'sex', 'country'])

# Prepare features
X, y = prepare_features_for_modeling(fraud_df, target_col='class')
print(f"   Features shape: {X.shape}")
print(f"   Target distribution: {dict(y.value_counts())}")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# ============================================================================
# 3. APPLY SMOTE
# ============================================================================
print("\n3. Applying SMOTE to training data...")
X_train_smote, y_train_smote = apply_smote(X_train, y_train, sampling_strategy=0.5)
print(f"   Before SMOTE: {dict(y_train.value_counts())}")
print(f"   After SMOTE: {dict(pd.Series(y_train_smote).value_counts())}")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n4. Training models...")
models = {}

# 4.1 Logistic Regression (Baseline)
print("   Training Logistic Regression...")
lr_model = train_logistic_regression(X_train_smote, y_train_smote)
models['Logistic Regression'] = lr_model
print("   ✓ Logistic Regression trained")

# 4.2 Random Forest
print("   Training Random Forest...")
rf_model = train_random_forest(X_train_smote, y_train_smote, n_estimators=100, max_depth=10)
models['Random Forest'] = rf_model
print("   ✓ Random Forest trained")

# 4.3 XGBoost
if XGBOOST_AVAILABLE:
    print("   Training XGBoost...")
    xgb_model = train_xgboost(X_train_smote, y_train_smote, n_estimators=100, max_depth=6)
    models['XGBoost'] = xgb_model
    print("   ✓ XGBoost trained")
else:
    print("   ⚠ XGBoost not available")

# 4.4 LightGBM
if LIGHTGBM_AVAILABLE:
    print("   Training LightGBM...")
    lgb_model = train_lightgbm(X_train_smote, y_train_smote, n_estimators=100, max_depth=6)
    models['LightGBM'] = lgb_model
    print("   ✓ LightGBM trained")
else:
    print("   ⚠ LightGBM not available")

# ============================================================================
# 5. EVALUATE MODELS
# ============================================================================
print("\n5. Evaluating models...")
results = {}
for name, model in models.items():
    metrics = evaluate_model(model, X_test, y_test)
    results[name] = {
        'Precision': round(metrics['precision'], 4),
        'Recall': round(metrics['recall'], 4),
        'F1-Score': round(metrics['f1_score'], 4),
        'ROC-AUC': round(metrics['roc_auc'], 4),
        'AUC-PR': round(metrics['average_precision'], 4)
    }
    print(f"   {name}: F1={metrics['f1_score']:.4f}, AUC-PR={metrics['average_precision']:.4f}")

# Model comparison dataframe
comparison_df = compare_models(models, X_test, y_test)
print("\n" + "=" * 70)
print("MODEL COMPARISON (sorted by F1-Score)")
print("=" * 70)
print(comparison_df.to_string(index=False))

# ============================================================================
# 6. SELECT BEST MODEL
# ============================================================================
best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
best_metrics = results[best_model_name]

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print("=" * 70)
print(f"   Precision: {best_metrics['Precision']}")
print(f"   Recall: {best_metrics['Recall']}")
print(f"   F1-Score: {best_metrics['F1-Score']}")
print(f"   ROC-AUC: {best_metrics['ROC-AUC']}")
print(f"   AUC-PR: {best_metrics['AUC-PR']}")

# ============================================================================
# 7. SAVE MODELS
# ============================================================================
print("\n7. Saving models...")
os.makedirs('models', exist_ok=True)

for name, model in models.items():
    filename = f"models/{name.lower().replace(' ', '_')}_fraud_model.pkl"
    save_model(model, filename)

# Save the best model with a special name
best_filename = f"models/best_model.pkl"
save_model(best_model, best_filename)

# Save results to JSON
results_file = 'models/model_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'best_model': best_model_name,
        'results': results,
        'feature_names': X.columns.tolist(),
        'trained_at': datetime.now().isoformat()
    }, f, indent=2)
print(f"   Results saved to {results_file}")

# ============================================================================
# 8. GENERATE VISUALIZATION
# ============================================================================
print("\n8. Generating visualizations...")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'AUC-PR']
x = np.arange(len(comparison_df))
width = 0.15

for i, metric in enumerate(metrics_names):
    ax.bar(x + i*width, comparison_df[metric], width, label=metric)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison - E-commerce Fraud Detection', fontsize=14)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(comparison_df['Model'])
ax.legend(loc='lower right')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Model comparison chart saved")

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    if idx >= 4:
        break
    metrics = evaluate_model(model, X_test, y_test)
    cm = metrics['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'{name}\n(F1={results[name]["F1-Score"]:.4f})')

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
print("   ✓ Confusion matrices saved")

# Feature importance (from best model if tree-based)
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top 15 Feature Importances ({best_model_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
    print("   ✓ Feature importance chart saved")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nArtifacts saved:")
print(f"   - Models: models/*.pkl")
print(f"   - Results: models/model_results.json")
print(f"   - Figures: figures/model_comparison.png")
print(f"   - Figures: figures/confusion_matrices.png")
print(f"   - Figures: figures/feature_importance.png")

