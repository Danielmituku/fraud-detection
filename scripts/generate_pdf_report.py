"""
Generate PDF Report from Markdown
Uses markdown2 and weasyprint for conversion
"""

import os
import sys

def generate_pdf_simple():
    """Generate a simple PDF summary using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import json
    
    # Load results
    with open('models/final_model_results.json', 'r') as f:
        results = json.load(f)
    
    # Create PDF
    with PdfPages('reports/final-report.pdf') as pdf:
        # Title Page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        title_text = """
FRAUD DETECTION SYSTEM
Final Report

Improved Detection of Fraud Cases for
E-commerce and Bank Transactions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Author: Daniel Mituku
Organization: Adey Innovations Inc.
Date: December 30, 2025

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

10Academy Week 5 Challenge
        """
        ax.text(0.5, 0.5, title_text, ha='center', va='center', 
                fontsize=14, family='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Executive Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        summary_text = """
EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This project developed fraud detection models for:
• E-commerce transactions (151,112 records)
• Bank credit card transactions (284,807 records)

KEY ACHIEVEMENTS:
━━━━━━━━━━━━━━━━━━
✓ Processed 435,000+ transactions across two datasets
✓ Engineered 12+ fraud indicator features
✓ Implemented cross-validation and hyperparameter tuning
✓ Generated SHAP explainability analysis
✓ Provided actionable business recommendations

FINAL RESULTS:
━━━━━━━━━━━━━━
┌─────────────────┬───────────────────────┬─────────┬──────────┐
│ Dataset         │ Best Model            │ AUC-PR  │ F1-Score │
├─────────────────┼───────────────────────┼─────────┼──────────┤
│ E-commerce      │ Random Forest (Tuned) │ 0.7126  │ 0.6277   │
│ Credit Card     │ Gradient Boosting     │ 0.8583  │ 0.7685   │
└─────────────────┴───────────────────────┴─────────┴──────────┘

BUSINESS IMPACT:
━━━━━━━━━━━━━━━━
• E-commerce: 68.2% of fraud detected
• Credit Card: 84.7% of fraud detected with 70% precision
        """
        ax.text(0.05, 0.95, summary_text, ha='left', va='top', 
                fontsize=10, family='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # E-commerce Results
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ecom_text = f"""
E-COMMERCE FRAUD DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: Fraud_Data.csv (151,112 transactions, 9.36% fraud rate)

CROSS-VALIDATION RESULTS (5-Fold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model                  F1-Score (mean±std)    ROC-AUC (mean±std)
──────────────────────────────────────────────────────────────────
Gradient Boosting      0.9254 ± 0.0023        0.9672 ± 0.0015
Random Forest          0.8323 ± 0.0027        0.9539 ± 0.0016
Logistic Regression    0.7369 ± 0.0046        0.8679 ± 0.0014

TEST SET RESULTS:
━━━━━━━━━━━━━━━━
Model                  Precision  Recall   F1-Score  AUC-PR
──────────────────────────────────────────────────────────────────
Random Forest (Tuned)  0.5815     0.6820   0.6277    0.7126  ⭐
Gradient Boosting      0.9987     0.5406   0.7015    0.7117
Logistic Regression    0.5326     0.6982   0.6043    0.6643

BEST MODEL: Random Forest (Tuned)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Hyperparameters: max_depth=12, n_estimators=100
• Selected for: Highest AUC-PR with balanced recall
• Fraud Detection Rate: 68.2%

TOP FEATURES (SHAP Analysis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. is_weekend (0.0995)
2. day_of_week (0.0458)
3. time_since_signup (0.0391)
4. device_unique_users (0.0376)
5. signup_day_of_week (0.0369)
        """
        ax.text(0.05, 0.95, ecom_text, ha='left', va='top', 
                fontsize=9, family='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Credit Card Results
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        cc_text = """
CREDIT CARD FRAUD DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: creditcard.csv (284,807 transactions, 0.17% fraud rate)
Note: Extreme class imbalance (578:1 ratio)

CROSS-VALIDATION RESULTS (5-Fold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model                  F1-Score (mean±std)    ROC-AUC (mean±std)
──────────────────────────────────────────────────────────────────
Random Forest          0.7139 ± 0.1071        0.9294 ± 0.0200
Gradient Boosting      0.6070 ± 0.1260        0.8409 ± 0.1254
Logistic Regression    0.0858 ± 0.0058        0.9251 ± 0.0461

TEST SET RESULTS:
━━━━━━━━━━━━━━━━
Model                  Precision  Recall   F1-Score  AUC-PR
──────────────────────────────────────────────────────────────────
Gradient Boosting      0.7034     0.8469   0.7685    0.8583  ⭐
Random Forest (Tuned)  0.3257     0.8673   0.4735    0.7773
Logistic Regression    0.0580     0.9184   0.1092    0.7256

BEST MODEL: Gradient Boosting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Configuration: n_estimators=100, max_depth=6
• Selected for: Best balance of precision and recall
• Fraud Detection Rate: 84.7%
• Precision: 70.3% (low false alarm rate)

TOP FEATURES (PCA-transformed):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V14, V17, V12, V10, V3, Amount, Time
(Features are anonymized via PCA)
        """
        ax.text(0.05, 0.95, cc_text, ha='left', va='top', 
                fontsize=9, family='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Business Recommendations
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        rec_text = """
BUSINESS RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on SHAP analysis and model insights:

1. TIME-BASED VERIFICATION (CRITICAL)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Finding: 99.52% of transactions within 1 hour of signup are fraudulent
   
   Recommendation:
   • Purchase < 1 hour after signup → Require phone verification (SMS OTP)
   • Purchase < 24 hours after signup → Email confirmation required
   • First high-value purchase → Manual review queue
   
   Expected Impact: Prevent 30-40% of fraudulent transactions


2. DEVICE INTELLIGENCE
   ━━━━━━━━━━━━━━━━━━━━
   Finding: Devices with multiple users indicate fraud rings
   
   Recommendation:
   • Device with 3+ users → Automatic alert
   • Known bad device → Block transaction
   • New device on established account → Step-up authentication
   
   Expected Impact: Detect organized fraud rings


3. TRANSACTION VELOCITY CONTROLS
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Finding: Transaction frequency patterns distinguish fraud
   
   Recommendation:
   • > 3 transactions/hour (new account) → Soft block + verification
   • Unusual pattern vs. history → Real-time alert
   
   Expected Impact: Reduce automated fraud attacks


4. GEOGRAPHIC RISK SCORING
   ━━━━━━━━━━━━━━━━━━━━━━━━
   Finding: Country-based features contribute to fraud prediction
   
   Recommendation:
   • High-risk country IP → Enhanced verification
   • IP/location mismatch → Flag for review
   
   Expected Impact: Identify cross-border fraud
        """
        ax.text(0.05, 0.95, rec_text, ha='left', va='top', 
                fontsize=9, family='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Include figures
        figure_files = [
            ('figures/class_imbalance_comparison.png', 'Class Imbalance Analysis'),
            ('figures/cross_validation_comparison.png', 'Cross-Validation Results'),
            ('figures/final_test_comparison.png', 'Final Model Comparison'),
            ('figures/final_confusion_matrices.png', 'Confusion Matrices'),
            ('figures/shap_summary_plot.png', 'SHAP Feature Importance'),
            ('figures/feature_importance.png', 'Model Feature Importance'),
        ]
        
        for fig_path, title in figure_files:
            if os.path.exists(fig_path):
                fig, ax = plt.subplots(figsize=(11, 8.5))
                img = plt.imread(fig_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Conclusion
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        conclusion_text = """
CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY ACHIEVEMENTS:
━━━━━━━━━━━━━━━━

✓ Task 1 - Data Analysis & Preprocessing
  • Cleaned and analyzed 435,000+ transactions
  • Integrated geolocation data (IP to Country)
  • Engineered 12+ meaningful features
  • Addressed severe class imbalance with SMOTE

✓ Task 2 - Model Building & Training
  • Trained 3 model types on 2 datasets
  • Implemented Stratified 5-Fold Cross-Validation
  • Applied StandardScaler normalization
  • Performed GridSearchCV tuning
  • E-commerce: 71.26% AUC-PR
  • Credit Card: 85.83% AUC-PR

✓ Task 3 - Model Explainability
  • Generated SHAP summary and force plots
  • Identified top fraud drivers
  • Provided 4 actionable business recommendations


FUTURE WORK:
━━━━━━━━━━━━
• Deploy models as REST API (Flask/FastAPI)
• Build real-time monitoring dashboard
• Implement adaptive thresholds
• Add neural network models


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Report Prepared By: Daniel Mituku
Organization: Adey Innovations Inc.
Date: December 30, 2025

GitHub: https://github.com/Danielmituku/fraud-detection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        ax.text(0.05, 0.95, conclusion_text, ha='left', va='top', 
                fontsize=9, family='monospace', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print("✓ PDF report generated: reports/final-report.pdf")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_pdf_simple()

