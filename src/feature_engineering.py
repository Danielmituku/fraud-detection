"""
Feature Engineering Module

This module provides functions for creating features from fraud detection data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from purchase_time.
    
    Features created:
    - hour_of_day: Hour when transaction occurred (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_weekend: Boolean flag for weekend transactions
    - time_since_signup: Seconds between signup and purchase
    
    Parameters
    ----------
    df : pd.DataFrame
        Fraud data with signup_time and purchase_time columns
        
    Returns
    -------
    pd.DataFrame
        Data with time features added
    """
    df = df.copy()
    
    # Extract hour and day
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Time since signup (in seconds)
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds()
    
    # Additional time features
    df['signup_hour'] = df['signup_time'].dt.hour
    df['signup_day_of_week'] = df['signup_time'].dt.dayofweek
    
    return df


def create_transaction_velocity_features(df: pd.DataFrame, 
                                         time_windows: List[int] = [3600, 86400, 604800]) -> pd.DataFrame:
    """
    Create transaction velocity features.
    
    Calculates the number of transactions per user within various time windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Fraud data with user_id and purchase_time
    time_windows : List[int]
        Time windows in seconds (default: 1 hour, 1 day, 1 week)
        
    Returns
    -------
    pd.DataFrame
        Data with velocity features added
    """
    df = df.copy()
    
    # Sort by user and time
    df = df.sort_values(['user_id', 'purchase_time'])
    
    # Count transactions per user
    user_tx_counts = df.groupby('user_id').size().reset_index(name='user_total_transactions')
    df = df.merge(user_tx_counts, on='user_id', how='left')
    
    # Transaction number for each user (cumulative count)
    df['user_transaction_number'] = df.groupby('user_id').cumcount() + 1
    
    return df


def create_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create device-related features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Fraud data with device_id column
        
    Returns
    -------
    pd.DataFrame
        Data with device features added
    """
    df = df.copy()
    
    # Count transactions per device
    device_tx_counts = df.groupby('device_id').size().reset_index(name='device_total_transactions')
    df = df.merge(device_tx_counts, on='device_id', how='left')
    
    # Count unique users per device
    device_user_counts = df.groupby('device_id')['user_id'].nunique().reset_index(name='device_unique_users')
    df = df.merge(device_user_counts, on='device_id', how='left')
    
    return df


def encode_categorical_features(df: pd.DataFrame, 
                                categorical_cols: List[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using One-Hot Encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with categorical columns
    categorical_cols : List[str], optional
        List of categorical columns to encode
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Encoded data and dictionary of encodings
    """
    df = df.copy()
    
    if categorical_cols is None:
        categorical_cols = ['source', 'browser', 'sex']
    
    # Filter to existing columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Store mapping info
    encoding_info = {col: df[col].unique().tolist() for col in categorical_cols}
    
    return df_encoded, encoding_info


def scale_numerical_features(df: pd.DataFrame,
                            numerical_cols: List[str] = None,
                            scaler_type: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features using StandardScaler or MinMaxScaler.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with numerical columns
    numerical_cols : List[str], optional
        List of numerical columns to scale
    scaler_type : str
        Type of scaler ('standard' or 'minmax')
        
    Returns
    -------
    Tuple[pd.DataFrame, object]
        Scaled data and fitted scaler
    """
    df = df.copy()
    
    if numerical_cols is None:
        numerical_cols = ['purchase_value', 'age', 'time_since_signup']
    
    # Filter to existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Select scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Scale features
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler


def create_all_features(df: pd.DataFrame, 
                       scale: bool = True,
                       encode: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud data
    scale : bool
        Whether to scale numerical features
    encode : bool
        Whether to encode categorical features
        
    Returns
    -------
    pd.DataFrame
        Feature-engineered data
    """
    # Time features
    df = create_time_features(df)
    
    # Velocity features
    df = create_transaction_velocity_features(df)
    
    # Device features
    df = create_device_features(df)
    
    # Encode categorical
    if encode:
        df, _ = encode_categorical_features(df)
    
    # Scale numerical (do this after encoding)
    if scale:
        numerical_cols = ['purchase_value', 'age', 'time_since_signup', 
                         'hour_of_day', 'user_total_transactions',
                         'device_total_transactions', 'device_unique_users']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        df, _ = scale_numerical_features(df, numerical_cols)
    
    return df


def prepare_features_for_modeling(df: pd.DataFrame, 
                                  target_col: str = 'class',
                                  drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data
    target_col : str
        Name of target column
    drop_cols : List[str], optional
        Additional columns to drop
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (X) and target (y)
    """
    df = df.copy()
    
    # Default columns to drop
    if drop_cols is None:
        drop_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    
    # Filter to existing columns
    drop_cols = [col for col in drop_cols if col in df.columns]
    drop_cols.append(target_col)
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    return X, y

