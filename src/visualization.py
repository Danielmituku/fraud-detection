"""
Visualization Module

This module provides functions for creating visualizations for EDA and model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def plot_class_distribution(y: pd.Series,
                           title: str = 'Class Distribution',
                           figsize: Tuple[int, int] = (8, 5)) -> plt.Figure:
    """
    Plot the distribution of classes.
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    counts = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(counts.index.astype(str), counts.values, color=colors)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Counts')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
    
    # Add count labels
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=['Legitimate', 'Fraud'], 
                autopct='%1.2f%%', colors=colors, explode=[0, 0.1])
    axes[1].set_title('Class Percentages')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_numerical_distributions(df: pd.DataFrame,
                                 numerical_cols: List[str],
                                 target_col: str = 'class',
                                 figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot distributions of numerical features by class.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with numerical features
    numerical_cols : List[str]
        List of numerical columns to plot
    target_col : str
        Target column name
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
                subset = df[df[target_col] == label][col]
                axes[i].hist(subset, bins=50, alpha=0.7, label=f'Class {label}', color=color)
            
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].legend()
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_categorical_distributions(df: pd.DataFrame,
                                   categorical_cols: List[str],
                                   target_col: str = 'class',
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot distributions of categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with categorical features
    categorical_cols : List[str]
        List of categorical columns to plot
    target_col : str
        Target column name
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            # Calculate fraud rate by category
            fraud_rate = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fraud_rate)))
            
            axes[i].barh(range(len(fraud_rate)), fraud_rate.values, color=colors)
            axes[i].set_yticks(range(len(fraud_rate)))
            axes[i].set_yticklabels(fraud_rate.index)
            axes[i].set_xlabel('Fraud Rate')
            axes[i].set_title(f'Fraud Rate by {col}')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame,
                            figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with numerical features
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numerical_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_proba: np.ndarray,
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='#3498db', linewidth=2,
            label=f'PR Curve (AUC = {avg_precision:.3f})')
    ax.fill_between(recall, precision, alpha=0.3, color='#3498db')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='#e74c3c', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot model comparison bar chart.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model metrics
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'AUC-PR']
    metrics = [m for m in metrics if m in comparison_df.columns]
    
    x = np.arange(len(comparison_df))
    width = 0.15
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, comparison_df[metric], width, 
                     label=metric, color=colors[i])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_time_based_fraud(df: pd.DataFrame,
                          time_col: str = 'hour_of_day',
                          target_col: str = 'class',
                          figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot fraud rate by time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with time and target columns
    time_col : str
        Time column name
    target_col : str
        Target column name
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Fraud rate by hour
    fraud_by_time = df.groupby(time_col)[target_col].mean()
    
    axes[0].bar(fraud_by_time.index, fraud_by_time.values, color='#3498db')
    axes[0].set_xlabel(time_col.replace('_', ' ').title())
    axes[0].set_ylabel('Fraud Rate')
    axes[0].set_title(f'Fraud Rate by {time_col.replace("_", " ").title()}')
    
    # Transaction count by hour
    tx_by_time = df.groupby(time_col).size()
    
    axes[1].bar(tx_by_time.index, tx_by_time.values, color='#2ecc71')
    axes[1].set_xlabel(time_col.replace('_', ' ').title())
    axes[1].set_ylabel('Transaction Count')
    axes[1].set_title(f'Transaction Count by {time_col.replace("_", " ").title()}')
    
    plt.tight_layout()
    return fig


def plot_fraud_by_country(df: pd.DataFrame,
                          top_n: int = 15,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot fraud rate by country.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with country and class columns
    top_n : int
        Number of countries to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Filter to countries with sufficient transactions
    country_counts = df['country'].value_counts()
    valid_countries = country_counts[country_counts >= 100].index
    df_filtered = df[df['country'].isin(valid_countries)]
    
    # Top countries by fraud rate
    fraud_by_country = df_filtered.groupby('country')['class'].mean().sort_values(ascending=False)
    top_fraud = fraud_by_country.head(top_n)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_fraud)))
    axes[0].barh(range(len(top_fraud)), top_fraud.values, color=colors)
    axes[0].set_yticks(range(len(top_fraud)))
    axes[0].set_yticklabels(top_fraud.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Fraud Rate')
    axes[0].set_title(f'Top {top_n} Countries by Fraud Rate')
    
    # Top countries by transaction volume
    top_volume = country_counts.head(top_n)
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_volume)))
    axes[1].barh(range(len(top_volume)), top_volume.values, color=colors)
    axes[1].set_yticks(range(len(top_volume)))
    axes[1].set_yticklabels(top_volume.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Transaction Count')
    axes[1].set_title(f'Top {top_n} Countries by Volume')
    
    plt.tight_layout()
    return fig

