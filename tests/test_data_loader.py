"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import (
    ip_to_integer,
    clean_fraud_data,
    clean_creditcard_data,
    get_class_distribution
)


class TestIpToInteger:
    """Tests for IP address conversion."""
    
    def test_ip_to_integer_basic(self):
        """Test basic IP to integer conversion."""
        result = ip_to_integer(3232235777.0)  # 192.168.1.1
        assert result == 3232235777
    
    def test_ip_to_integer_zero(self):
        """Test conversion of zero IP."""
        result = ip_to_integer(0.0)
        assert result == 0
    
    def test_ip_to_integer_max(self):
        """Test conversion of maximum IP."""
        result = ip_to_integer(4294967295.0)  # 255.255.255.255
        assert result == 4294967295


class TestCleanFraudData:
    """Tests for fraud data cleaning."""
    
    @pytest.fixture
    def sample_fraud_data(self):
        """Create sample fraud data for testing."""
        return pd.DataFrame({
            'user_id': [1, 2, 3, 4, 4],
            'signup_time': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-04']),
            'purchase_time': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-05']),
            'purchase_value': [100.0, 200.0, 300.0, 400.0, 400.0],
            'device_id': ['d1', 'd2', 'd3', 'd4', 'd4'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO', 'SEO'],
            'browser': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Chrome'],
            'sex': ['M', 'F', 'M', 'F', 'F'],
            'age': [25, 30, 35, 40, 40],
            'ip_address': [1.0, 2.0, 3.0, 4.0, 4.0],
            'class': [0, 1, 0, 1, 1]
        })
    
    def test_clean_removes_duplicates(self, sample_fraud_data):
        """Test that duplicates are removed."""
        # Add a duplicate row
        sample_fraud_data = pd.concat([sample_fraud_data, sample_fraud_data.iloc[[0]]], ignore_index=True)
        cleaned = clean_fraud_data(sample_fraud_data)
        assert len(cleaned) < len(sample_fraud_data)
    
    def test_clean_handles_missing_values(self, sample_fraud_data):
        """Test handling of missing values."""
        sample_fraud_data.loc[0, 'purchase_value'] = np.nan
        cleaned = clean_fraud_data(sample_fraud_data)
        assert not cleaned['purchase_value'].isnull().any()


class TestCleanCreditcardData:
    """Tests for credit card data cleaning."""
    
    @pytest.fixture
    def sample_creditcard_data(self):
        """Create sample credit card data for testing."""
        return pd.DataFrame({
            'Time': [0, 1, 2, 3, 4],
            'V1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'V2': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Amount': [100.0, 200.0, 300.0, 400.0, 500.0],
            'Class': [0, 0, 1, 0, 0]
        })
    
    def test_clean_creditcard_preserves_data(self, sample_creditcard_data):
        """Test that clean data is preserved."""
        cleaned = clean_creditcard_data(sample_creditcard_data)
        assert len(cleaned) == len(sample_creditcard_data)


class TestGetClassDistribution:
    """Tests for class distribution calculation."""
    
    def test_get_class_distribution_basic(self):
        """Test basic class distribution calculation."""
        y = pd.Series([0, 0, 0, 0, 1])
        dist = get_class_distribution(pd.DataFrame({'class': y}))
        
        assert dist['counts'][0] == 4
        assert dist['counts'][1] == 1
        assert dist['percentages'][0] == 80.0
        assert dist['percentages'][1] == 20.0
        assert dist['imbalance_ratio'] == 4.0
    
    def test_get_class_distribution_custom_column(self):
        """Test with custom target column name."""
        df = pd.DataFrame({'target': [0, 0, 1, 1, 1]})
        dist = get_class_distribution(df, target_col='target')
        
        assert dist['counts'][0] == 2
        assert dist['counts'][1] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

