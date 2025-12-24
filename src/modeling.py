"""
Modeling Module

This module provides functions for training and evaluating fraud detection models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

# Optional imports for ensemble models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def stratified_train_test_split(X: pd.DataFrame, 
                                 y: pd.Series,
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple:
    """
    Perform stratified train-test split.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )


def apply_smote(X_train: pd.DataFrame, 
                y_train: pd.Series,
                random_state: int = 42,
                sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to handle class imbalance.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    random_state : int
        Random seed
    sampling_strategy : float
        Ratio of minority to majority class after resampling
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled X_train and y_train
    """
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def apply_undersampling(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        random_state: int = 42,
                        sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply random undersampling to handle class imbalance.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    random_state : int
        Random seed
    sampling_strategy : float
        Ratio of minority to majority class after resampling
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled X_train and y_train
    """
    undersampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              class_weight: str = 'balanced',
                              max_iter: int = 1000) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    class_weight : str
        Class weight strategy
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    LogisticRegression
        Trained model
    """
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        n_estimators: int = 100,
                        max_depth: int = 10,
                        class_weight: str = 'balanced') -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    class_weight : str
        Class weight strategy
        
    Returns
    -------
    RandomForestClassifier
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  n_estimators: int = 100,
                  max_depth: int = 6,
                  learning_rate: float = 0.1,
                  scale_pos_weight: float = None) -> Any:
    """
    Train an XGBoost model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
    scale_pos_weight : float
        Balance of positive and negative weights
        
    Returns
    -------
    XGBClassifier
        Trained model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    if scale_pos_weight is None:
        # Calculate based on class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   n_estimators: int = 100,
                   max_depth: int = 6,
                   learning_rate: float = 0.1,
                   class_weight: str = 'balanced') -> Any:
    """
    Train a LightGBM model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
    class_weight : str
        Class weight strategy
        
    Returns
    -------
    LGBMClassifier
        Trained model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
    
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'average_precision': average_precision_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics


def cross_validate_model(model: Any,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv: int = 5) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.
    
    Parameters
    ----------
    model : Any
        Model to evaluate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of folds
        
    Returns
    -------
    Dict[str, Any]
        Cross-validation results
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Define scorers
    scoring = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    
    results = {}
    for metric_name, scorer in scoring.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring=scorer)
        results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    return results


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Parameters
    ----------
    model : Any
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model
        
    Returns
    -------
    Any
        Loaded model
    """
    return joblib.load(filepath)


def compare_models(models: Dict[str, Any],
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of model name to trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns
    -------
    pd.DataFrame
        Comparison table of model metrics
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results.append({
            'Model': name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'AUC-PR': metrics['average_precision']
        })
    
    return pd.DataFrame(results).sort_values('F1-Score', ascending=False)

