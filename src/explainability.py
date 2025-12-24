"""
Model Explainability Module

This module provides functions for interpreting model predictions using SHAP.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_feature_importance(model: Any, 
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Extract built-in feature importance from tree-based models.
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : List[str]
        List of feature names
        
    Returns
    -------
    pd.DataFrame
        Feature importance ranking
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importance attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 10,
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot top N feature importances.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance data
    top_n : int
        Number of top features to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    plt.tight_layout()
    return fig


def create_shap_explainer(model: Any, 
                          X_train: pd.DataFrame,
                          model_type: str = 'tree') -> Any:
    """
    Create a SHAP explainer for the model.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_train : pd.DataFrame
        Training data for background
    model_type : str
        Type of model ('tree', 'linear', 'kernel')
        
    Returns
    -------
    Any
        SHAP explainer
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Use sampling for kernel explainer (slow for large datasets)
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    return explainer


def calculate_shap_values(explainer: Any, 
                          X: pd.DataFrame) -> np.ndarray:
    """
    Calculate SHAP values for the given data.
    
    Parameters
    ----------
    explainer : Any
        SHAP explainer
    X : pd.DataFrame
        Data to explain
        
    Returns
    -------
    np.ndarray
        SHAP values
    """
    shap_values = explainer.shap_values(X)
    
    # For binary classification, return values for positive class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return shap_values[1]
    
    return shap_values


def plot_shap_summary(shap_values: np.ndarray,
                      X: pd.DataFrame,
                      figsize: Tuple[int, int] = (10, 8),
                      max_display: int = 20) -> plt.Figure:
    """
    Create SHAP summary plot.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame
        Feature data
    figsize : Tuple[int, int]
        Figure size
    max_display : int
        Maximum features to display
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values: np.ndarray,
                  X: pd.DataFrame,
                  figsize: Tuple[int, int] = (10, 8),
                  max_display: int = 15) -> plt.Figure:
    """
    Create SHAP bar plot (global feature importance).
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame
        Feature data
    figsize : Tuple[int, int]
        Figure size
    max_display : int
        Maximum features to display
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X, plot_type='bar', max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_force_plot(explainer: Any,
                    shap_values: np.ndarray,
                    X: pd.DataFrame,
                    idx: int) -> Any:
    """
    Create a SHAP force plot for a single prediction.
    
    Parameters
    ----------
    explainer : Any
        SHAP explainer
    shap_values : np.ndarray
        SHAP values
    X : pd.DataFrame
        Feature data
    idx : int
        Index of the sample to explain
        
    Returns
    -------
    Any
        SHAP force plot
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    return shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[idx],
        X.iloc[idx]
    )


def find_prediction_examples(model: Any,
                             X: pd.DataFrame,
                             y: pd.Series) -> dict:
    """
    Find examples of true positives, false positives, and false negatives.
    
    Parameters
    ----------
    model : Any
        Trained model
    X : pd.DataFrame
        Features
    y : pd.Series
        True labels
        
    Returns
    -------
    dict
        Dictionary with indices of TP, FP, FN examples
    """
    y_pred = model.predict(X)
    
    # True Positives: actual=1, predicted=1
    tp_mask = (y.values == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    
    # False Positives: actual=0, predicted=1
    fp_mask = (y.values == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    
    # False Negatives: actual=1, predicted=0
    fn_mask = (y.values == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    
    return {
        'true_positives': tp_indices.tolist() if len(tp_indices) > 0 else [],
        'false_positives': fp_indices.tolist() if len(fp_indices) > 0 else [],
        'false_negatives': fn_indices.tolist() if len(fn_indices) > 0 else []
    }


def get_top_shap_features(shap_values: np.ndarray,
                          feature_names: List[str],
                          top_n: int = 5) -> pd.DataFrame:
    """
    Get the top N features by mean absolute SHAP value.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    feature_names : List[str]
        Feature names
    top_n : int
        Number of top features
        
    Returns
    -------
    pd.DataFrame
        Top features and their importance
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    return importance_df.head(top_n)


def compare_feature_importance(model_importance: pd.DataFrame,
                               shap_importance: pd.DataFrame) -> pd.DataFrame:
    """
    Compare model's built-in feature importance with SHAP importance.
    
    Parameters
    ----------
    model_importance : pd.DataFrame
        Feature importance from model
    shap_importance : pd.DataFrame
        Feature importance from SHAP
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    # Merge on feature name
    comparison = model_importance.merge(
        shap_importance,
        on='feature',
        how='outer',
        suffixes=('_model', '_shap')
    )
    
    # Rank features
    comparison['rank_model'] = comparison['importance'].rank(ascending=False)
    comparison['rank_shap'] = comparison['mean_abs_shap'].rank(ascending=False)
    comparison['rank_diff'] = abs(comparison['rank_model'] - comparison['rank_shap'])
    
    return comparison.sort_values('rank_shap')

