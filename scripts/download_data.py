#!/usr/bin/env python3
"""
Data Download Script

This script downloads the required datasets for the fraud detection project.
Datasets are downloaded from Kaggle and saved to the data/raw directory.

Prerequisites:
- Kaggle API credentials configured (~/.kaggle/kaggle.json)
- kaggle package installed (pip install kaggle)

Usage:
    python scripts/download_data.py
"""

import os
import sys
from pathlib import Path
import zipfile
import shutil

# Try to import kaggle
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


def setup_directories():
    """Create necessary directories."""
    base_path = Path(__file__).parent.parent
    raw_data_path = base_path / 'data' / 'raw'
    raw_data_path.mkdir(parents=True, exist_ok=True)
    return raw_data_path


def download_fraud_ecommerce_data(output_path: Path):
    """
    Download the Fraud E-commerce dataset from Kaggle.
    
    Dataset: https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce
    Files: Fraud_Data.csv, IpAddress_to_Country.csv
    """
    if not KAGGLE_AVAILABLE:
        print("Kaggle API not available. Please install: pip install kaggle")
        print("And configure credentials: ~/.kaggle/kaggle.json")
        return False
    
    try:
        print("Downloading Fraud E-commerce dataset...")
        kaggle.api.dataset_download_files(
            'vbinh002/fraud-ecommerce',
            path=str(output_path),
            unzip=True
        )
        print("✓ Fraud E-commerce dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading Fraud E-commerce dataset: {e}")
        return False


def download_creditcard_data(output_path: Path):
    """
    Download the Credit Card Fraud dataset from Kaggle.
    
    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Files: creditcard.csv
    """
    if not KAGGLE_AVAILABLE:
        print("Kaggle API not available. Please install: pip install kaggle")
        return False
    
    try:
        print("Downloading Credit Card Fraud dataset...")
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(output_path),
            unzip=True
        )
        print("✓ Credit Card Fraud dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading Credit Card Fraud dataset: {e}")
        return False


def verify_datasets(data_path: Path):
    """Verify that all required datasets are present."""
    required_files = [
        'Fraud_Data.csv',
        'IpAddress_to_Country.csv',
        'creditcard.csv'
    ]
    
    print("\nVerifying datasets...")
    all_present = True
    
    for filename in required_files:
        filepath = data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            all_present = False
    
    return all_present


def print_manual_download_instructions():
    """Print instructions for manual download."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
If automatic download fails, please download manually:

1. Fraud E-commerce Dataset:
   URL: https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce
   Files needed: Fraud_Data.csv, IpAddress_to_Country.csv

2. Credit Card Fraud Dataset:
   URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   File needed: creditcard.csv

After downloading, place all CSV files in:
   fraud-detection/data/raw/

Note: You need a Kaggle account to download these datasets.
""")
    print("="*60)


def main():
    """Main function to download all datasets."""
    print("="*60)
    print("FRAUD DETECTION - DATA DOWNLOAD SCRIPT")
    print("="*60)
    
    # Setup directories
    raw_data_path = setup_directories()
    print(f"\nData directory: {raw_data_path}")
    
    # Check if data already exists
    if verify_datasets(raw_data_path):
        print("\n✓ All datasets already present!")
        return 0
    
    # Try to download datasets
    success = True
    
    if not (raw_data_path / 'Fraud_Data.csv').exists():
        if not download_fraud_ecommerce_data(raw_data_path):
            success = False
    
    if not (raw_data_path / 'creditcard.csv').exists():
        if not download_creditcard_data(raw_data_path):
            success = False
    
    # Verify final state
    print("\n" + "-"*40)
    if verify_datasets(raw_data_path):
        print("\n✓ All datasets downloaded successfully!")
        return 0
    else:
        print_manual_download_instructions()
        return 1


if __name__ == '__main__':
    sys.exit(main())

