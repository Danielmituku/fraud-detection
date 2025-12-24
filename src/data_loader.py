"""
Data Loading and Preprocessing Utilities

This module provides functions for loading, cleaning, and preprocessing
fraud detection datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def load_fraud_data(data_path: str = 'data/raw/Fraud_Data.csv') -> pd.DataFrame:
    """
    Load the e-commerce fraud dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the Fraud_Data.csv file
        
    Returns
    -------
    pd.DataFrame
        Loaded fraud data
    """
    df = pd.read_csv(data_path)
    
    # Convert timestamp columns to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    return df


def load_ip_to_country(data_path: str = 'data/raw/IpAddress_to_Country.csv') -> pd.DataFrame:
    """
    Load the IP address to country mapping dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the IpAddress_to_Country.csv file
        
    Returns
    -------
    pd.DataFrame
        Loaded IP to country mapping data
    """
    df = pd.read_csv(data_path)
    return df


def load_creditcard_data(data_path: str = 'data/raw/creditcard.csv') -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the creditcard.csv file
        
    Returns
    -------
    pd.DataFrame
        Loaded credit card data
    """
    df = pd.read_csv(data_path)
    return df


def ip_to_integer(ip_address: float) -> int:
    """
    Convert an IP address (as float) to integer format.
    
    The IP address in the dataset is stored as a float representation.
    This function converts it to an integer for range-based lookup.
    
    Parameters
    ----------
    ip_address : float
        IP address as a float
        
    Returns
    -------
    int
        IP address as integer
    """
    return int(ip_address)


def map_ip_to_country(fraud_df: pd.DataFrame, 
                      ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map IP addresses to countries using range-based lookup.
    
    Parameters
    ----------
    fraud_df : pd.DataFrame
        Fraud data with 'ip_address' column
    ip_country_df : pd.DataFrame
        IP to country mapping with lower_bound and upper_bound columns
        
    Returns
    -------
    pd.DataFrame
        Fraud data with 'country' column added
    """
    # Convert IP addresses to integers
    fraud_df = fraud_df.copy()
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_integer)
    
    # Sort IP ranges for merge_asof
    ip_country_df = ip_country_df.copy()
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    
    # Use merge_asof for efficient range-based lookup
    fraud_df = fraud_df.sort_values('ip_int')
    
    result = pd.merge_asof(
        fraud_df,
        ip_country_df[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Validate that IP is within range
    result.loc[result['ip_int'] > result['upper_bound_ip_address'], 'country'] = 'Unknown'
    
    # Clean up temporary columns
    result = result.drop(columns=['ip_int', 'lower_bound_ip_address', 'upper_bound_ip_address'])
    result = result.sort_index()
    
    return result


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the fraud dataset.
    
    Handles:
    - Missing values
    - Duplicates
    - Data type corrections
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud data
        
    Returns
    -------
    pd.DataFrame
        Cleaned fraud data
    """
    df = df.copy()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values per column:\n{missing[missing > 0]}")
        # Drop rows with missing values for key columns
        df = df.dropna(subset=['user_id', 'purchase_value', 'class'])
    
    # Ensure correct data types
    df['purchase_value'] = df['purchase_value'].astype(float)
    df['age'] = df['age'].astype(int)
    
    return df


def clean_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the credit card dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw credit card data
        
    Returns
    -------
    pd.DataFrame
        Cleaned credit card data
    """
    df = df.copy()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values per column:\n{missing[missing > 0]}")
        df = df.dropna()
    
    return df


def get_class_distribution(df: pd.DataFrame, target_col: str = 'class') -> dict:
    """
    Get the distribution of classes in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with target column
    target_col : str
        Name of the target column
        
    Returns
    -------
    dict
        Dictionary with class counts and percentages
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    return {
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict(),
        'imbalance_ratio': counts[0] / counts[1] if 1 in counts.index else None
    }

