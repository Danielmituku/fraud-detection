#!/usr/bin/env python3
"""
Generate all figures for the Interim-1 Report
This script runs the EDA and generates visualization outputs
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create figures directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def save_figure(fig, filename):
    """Save figure to figures directory"""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {filename}")

def load_fraud_data():
    """Load and prepare fraud dataset"""
    filepath = os.path.join(DATA_DIR, 'Fraud_Data.csv')
    df = pd.read_csv(filepath, parse_dates=['signup_time', 'purchase_time'])
    return df

def load_creditcard_data():
    """Load credit card dataset"""
    filepath = os.path.join(DATA_DIR, 'creditcard.csv')
    df = pd.read_csv(filepath)
    return df

def load_ip_to_country():
    """Load IP to country mapping"""
    filepath = os.path.join(DATA_DIR, 'IpAddress_to_Country.csv')
    df = pd.read_csv(filepath)
    return df

# ============================================
# FIGURE 1: Class Distribution - Fraud Data
# ============================================
def fig_fraud_class_distribution(df):
    """Generate class distribution plot for fraud data"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    class_counts = df['class'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    bars = axes[0].bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values, color=colors, edgecolor='black')
    axes[0].set_title('E-commerce Transaction Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Transactions', fontsize=12)
    axes[0].set_xlabel('Class', fontsize=12)
    
    # Add count labels
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                     f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.2f%%',
                colors=colors, explode=[0, 0.1], shadow=True, startangle=90)
    axes[1].set_title('Class Distribution Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'fraud_class_distribution.png')
    
    return {
        'legitimate': class_counts[0],
        'fraud': class_counts[1],
        'fraud_pct': class_counts[1] / len(df) * 100,
        'imbalance_ratio': class_counts[0] / class_counts[1]
    }

# ============================================
# FIGURE 2: Class Distribution - Credit Card
# ============================================
def fig_creditcard_class_distribution(df):
    """Generate class distribution plot for credit card data"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_counts = df['Class'].value_counts().sort_index()
    colors = ['#3498db', '#e74c3c']
    
    # Bar plot
    bars = axes[0].bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Credit Card Transaction Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Transactions', fontsize=12)
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_yscale('log')  # Log scale due to extreme imbalance
    
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                     f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.4f%%',
                colors=colors, explode=[0, 0.3], shadow=True, startangle=90)
    axes[1].set_title('Class Distribution (Note: Extreme Imbalance)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'creditcard_class_distribution.png')
    
    return {
        'legitimate': class_counts[0],
        'fraud': class_counts[1],
        'fraud_pct': class_counts[1] / len(df) * 100,
        'imbalance_ratio': class_counts[0] / class_counts[1]
    }

# ============================================
# FIGURE 3: Purchase Value Distribution
# ============================================
def fig_purchase_value_distribution(df):
    """Generate purchase value distribution by class"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    for cls, color, label in [(0, '#2ecc71', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['class'] == cls]['purchase_value']
        axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
    
    axes[0].set_title('Purchase Value Distribution by Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Purchase Value ($)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()
    
    # Box plot
    df.boxplot(column='purchase_value', by='class', ax=axes[1])
    axes[1].set_title('Purchase Value by Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=12)
    axes[1].set_ylabel('Purchase Value ($)', fontsize=12)
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    save_figure(fig, 'purchase_value_distribution.png')

# ============================================
# FIGURE 4: Age Distribution
# ============================================
def fig_age_distribution(df):
    """Generate age distribution by class"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls, color, label in [(0, '#2ecc71', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['class'] == cls]['age']
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
    
    ax.set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'age_distribution.png')

# ============================================
# FIGURE 5: Categorical Features Fraud Rates
# ============================================
def fig_categorical_fraud_rates(df):
    """Generate fraud rates by categorical features"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    categorical_cols = ['source', 'browser', 'sex']
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx // 2, idx % 2]
        
        # Calculate fraud rate per category
        fraud_rate = df.groupby(col)['class'].mean() * 100
        
        bars = ax.bar(fraud_rate.index, fraud_rate.values, color='#3498db', edgecolor='black')
        ax.axhline(y=df['class'].mean() * 100, color='red', linestyle='--', label='Overall Fraud Rate')
        ax.set_title(f'Fraud Rate by {col.title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraud Rate (%)', fontsize=10)
        ax.set_xlabel(col.title(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # Add value labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Remove unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'categorical_fraud_rates.png')

# ============================================
# FIGURE 6: Time Since Signup Analysis (KEY)
# ============================================
def fig_time_since_signup(df):
    """Generate time since signup analysis - KEY INSIGHT"""
    # Calculate time since signup
    df = df.copy()
    df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Create time buckets
    def categorize_time(hours):
        if hours < 1:
            return '< 1 hour'
        elif hours < 24:
            return '1-24 hours'
        elif hours < 168:
            return '1-7 days'
        elif hours < 720:
            return '1-4 weeks'
        else:
            return '> 1 month'
    
    df['time_bucket'] = df['time_since_signup_hours'].apply(categorize_time)
    
    # Order buckets
    bucket_order = ['< 1 hour', '1-24 hours', '1-7 days', '1-4 weeks', '> 1 month']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fraud rate by time bucket
    fraud_rate = df.groupby('time_bucket')['class'].mean() * 100
    fraud_rate = fraud_rate.reindex(bucket_order)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bucket_order)))
    bars = axes[0].bar(bucket_order, fraud_rate.values, color=colors, edgecolor='black')
    axes[0].axhline(y=df['class'].mean() * 100, color='blue', linestyle='--', label='Overall Rate')
    axes[0].set_title('🚨 Fraud Rate by Time Since Signup (KEY INSIGHT)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Fraud Rate (%)', fontsize=11)
    axes[0].set_xlabel('Time Since Signup', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Transaction count by time bucket and class
    counts = df.groupby(['time_bucket', 'class']).size().unstack(fill_value=0)
    counts = counts.reindex(bucket_order)
    counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1].set_title('Transaction Count by Time Bucket', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_xlabel('Time Since Signup', fontsize=11)
    axes[1].legend(['Legitimate', 'Fraud'])
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_figure(fig, 'time_since_signup_analysis.png')
    
    return fraud_rate.to_dict()

# ============================================
# FIGURE 7: Hour of Day Analysis
# ============================================
def fig_hour_of_day(df):
    """Generate hourly fraud analysis"""
    df = df.copy()
    df['hour'] = df['purchase_time'].dt.hour
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculate fraud rate per hour
    fraud_rate = df.groupby('hour')['class'].mean() * 100
    
    ax.plot(fraud_rate.index, fraud_rate.values, marker='o', linewidth=2, 
            markersize=8, color='#e74c3c', label='Fraud Rate')
    ax.fill_between(fraud_rate.index, fraud_rate.values, alpha=0.3, color='#e74c3c')
    ax.axhline(y=df['class'].mean() * 100, color='blue', linestyle='--', label='Overall Rate')
    
    ax.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day (0-23)', fontsize=12)
    ax.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'hour_of_day_analysis.png')

# ============================================
# FIGURE 8: Credit Card Amount Distribution
# ============================================
def fig_creditcard_amount(df):
    """Generate credit card amount analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Amount distribution (log scale)
    for cls, color, label in [(0, '#3498db', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['Class'] == cls]['Amount']
        axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
    
    axes[0].set_title('Transaction Amount Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Amount ($)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_xscale('symlog')
    axes[0].legend()
    
    # Box plot
    df.boxplot(column='Amount', by='Class', ax=axes[1])
    axes[1].set_title('Amount by Class', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=11)
    axes[1].set_ylabel('Amount ($)', fontsize=11)
    plt.suptitle('')
    
    plt.tight_layout()
    save_figure(fig, 'creditcard_amount_distribution.png')

# ============================================
# FIGURE 9: V-Features Correlation Heatmap
# ============================================
def fig_v_features_heatmap(df):
    """Generate correlation heatmap for top V-features"""
    # Find features with highest class separation
    v_cols = [f'V{i}' for i in range(1, 29)]
    
    fraud_mean = df[df['Class'] == 1][v_cols].mean()
    legit_mean = df[df['Class'] == 0][v_cols].mean()
    mean_diff = (fraud_mean - legit_mean).abs().sort_values(ascending=False)
    
    top_features = mean_diff.head(10).index.tolist()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df[top_features + ['Class']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                center=0, linewidths=0.5)
    ax.set_title('Correlation Matrix: Top 10 V-Features with Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'v_features_correlation.png')
    
    return top_features

# ============================================
# FIGURE 10: Class Imbalance Comparison
# ============================================
def fig_class_imbalance_comparison(fraud_stats, cc_stats):
    """Compare class imbalance between datasets"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['E-commerce\n(Fraud_Data)', 'Credit Card\n(creditcard)']
    fraud_rates = [fraud_stats['fraud_pct'], cc_stats['fraud_pct']]
    imbalance_ratios = [fraud_stats['imbalance_ratio'], cc_stats['imbalance_ratio']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars = ax.bar(x, fraud_rates, width, label='Fraud Rate (%)', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax.set_title('Class Imbalance Comparison Between Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    
    # Add annotations
    for i, (bar, ratio) in enumerate(zip(bars, imbalance_ratios)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{bar.get_height():.2f}%\n(Ratio: {ratio:.0f}:1)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, max(fraud_rates) * 1.5)
    
    plt.tight_layout()
    save_figure(fig, 'class_imbalance_comparison.png')

# ============================================
# FIGURE 11: Data Summary Statistics
# ============================================
def fig_data_summary(fraud_df, cc_df):
    """Create data summary visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dataset sizes
    sizes = [len(fraud_df), len(cc_df)]
    labels = ['E-commerce\n(Fraud_Data)', 'Credit Card\n(creditcard)']
    colors = ['#3498db', '#9b59b6']
    
    bars = axes[0].bar(labels, sizes, color=colors, edgecolor='black')
    axes[0].set_title('Dataset Sizes', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Transactions', fontsize=12)
    
    for bar, size in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                    f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Feature counts
    feature_counts = [fraud_df.shape[1], cc_df.shape[1]]
    bars2 = axes[1].bar(labels, feature_counts, color=colors, edgecolor='black')
    axes[1].set_title('Number of Features', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Feature Count', fontsize=12)
    
    for bar, count in zip(bars2, feature_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'data_summary.png')

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("=" * 60)
    print("FRAUD DETECTION - FIGURE GENERATION")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading datasets...")
    fraud_df = load_fraud_data()
    print(f"  ✓ Fraud_Data: {fraud_df.shape[0]:,} rows, {fraud_df.shape[1]} columns")
    
    cc_df = load_creditcard_data()
    print(f"  ✓ creditcard: {cc_df.shape[0]:,} rows, {cc_df.shape[1]} columns")
    
    ip_df = load_ip_to_country()
    print(f"  ✓ IpAddress_to_Country: {ip_df.shape[0]:,} rows")
    print()
    
    # Generate figures
    print("Generating figures...")
    print("-" * 40)
    
    # Fraud data figures
    fraud_stats = fig_fraud_class_distribution(fraud_df)
    cc_stats = fig_creditcard_class_distribution(cc_df)
    fig_purchase_value_distribution(fraud_df)
    fig_age_distribution(fraud_df)
    fig_categorical_fraud_rates(fraud_df)
    time_fraud_rates = fig_time_since_signup(fraud_df)
    fig_hour_of_day(fraud_df)
    
    # Credit card figures
    fig_creditcard_amount(cc_df)
    top_v_features = fig_v_features_heatmap(cc_df)
    
    # Comparison figures
    fig_class_imbalance_comparison(fraud_stats, cc_stats)
    fig_data_summary(fraud_df, cc_df)
    
    print()
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print()
    print("E-COMMERCE FRAUD DATA:")
    print(f"  Total transactions: {len(fraud_df):,}")
    print(f"  Legitimate: {fraud_stats['legitimate']:,} ({100-fraud_stats['fraud_pct']:.2f}%)")
    print(f"  Fraud: {fraud_stats['fraud']:,} ({fraud_stats['fraud_pct']:.2f}%)")
    print(f"  Imbalance ratio: {fraud_stats['imbalance_ratio']:.1f}:1")
    print()
    print("CREDIT CARD FRAUD DATA:")
    print(f"  Total transactions: {len(cc_df):,}")
    print(f"  Legitimate: {cc_stats['legitimate']:,} ({100-cc_stats['fraud_pct']:.4f}%)")
    print(f"  Fraud: {cc_stats['fraud']:,} ({cc_stats['fraud_pct']:.4f}%)")
    print(f"  Imbalance ratio: {cc_stats['imbalance_ratio']:.0f}:1")
    print()
    print("KEY INSIGHT - Time Since Signup Fraud Rates:")
    for bucket, rate in time_fraud_rates.items():
        print(f"  {bucket}: {rate:.2f}%")
    print()
    print("Top V-Features (Credit Card):", ', '.join(top_v_features[:5]))
    print()
    print(f"All figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

