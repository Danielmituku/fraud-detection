"""
Unit tests for feature_engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_engineering import (
    create_time_features,
    create_transaction_velocity_features,
    create_device_features,
    encode_categorical_features,
    scale_numerical_features,
    prepare_features_for_modeling
)


class TestTimeFeatures:
    """Tests for time feature creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'signup_time': pd.to_datetime(['2020-01-01 10:00:00', '2020-01-02 15:30:00']),
            'purchase_time': pd.to_datetime(['2020-01-01 12:00:00', '2020-01-03 20:00:00']),
            'user_id': [1, 2],
            'purchase_value': [100.0, 200.0]
        })
    
    def test_hour_of_day_extraction(self, sample_data):
        """Test hour extraction from purchase time."""
        result = create_time_features(sample_data)
        assert 'hour_of_day' in result.columns
        assert result['hour_of_day'].iloc[0] == 12
        assert result['hour_of_day'].iloc[1] == 20
    
    def test_day_of_week_extraction(self, sample_data):
        """Test day of week extraction."""
        result = create_time_features(sample_data)
        assert 'day_of_week' in result.columns
        # Wednesday = 2, Friday = 4 (0-indexed)
        assert result['day_of_week'].iloc[0] == 2  # Wednesday
    
    def test_time_since_signup(self, sample_data):
        """Test time since signup calculation."""
        result = create_time_features(sample_data)
        assert 'time_since_signup' in result.columns
        # First row: 2 hours = 7200 seconds
        assert result['time_since_signup'].iloc[0] == 7200.0
    
    def test_is_weekend_flag(self, sample_data):
        """Test weekend flag calculation."""
        result = create_time_features(sample_data)
        assert 'is_weekend' in result.columns


class TestVelocityFeatures:
    """Tests for transaction velocity features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple transactions per user."""
        return pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2],
            'purchase_time': pd.to_datetime([
                '2020-01-01 10:00:00',
                '2020-01-01 11:00:00',
                '2020-01-01 12:00:00',
                '2020-01-01 10:00:00',
                '2020-01-02 10:00:00'
            ])
        })
    
    def test_user_total_transactions(self, sample_data):
        """Test user total transactions count."""
        result = create_transaction_velocity_features(sample_data)
        assert 'user_total_transactions' in result.columns
        # User 1 has 3 transactions, User 2 has 2
        user1_count = result[result['user_id'] == 1]['user_total_transactions'].iloc[0]
        user2_count = result[result['user_id'] == 2]['user_total_transactions'].iloc[0]
        assert user1_count == 3
        assert user2_count == 2


class TestDeviceFeatures:
    """Tests for device-related features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with device information."""
        return pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'device_id': ['d1', 'd1', 'd1', 'd2']
        })
    
    def test_device_total_transactions(self, sample_data):
        """Test device transaction count."""
        result = create_device_features(sample_data)
        assert 'device_total_transactions' in result.columns
        # Device d1 has 3 transactions, d2 has 1
        d1_count = result[result['device_id'] == 'd1']['device_total_transactions'].iloc[0]
        d2_count = result[result['device_id'] == 'd2']['device_total_transactions'].iloc[0]
        assert d1_count == 3
        assert d2_count == 1
    
    def test_device_unique_users(self, sample_data):
        """Test unique users per device."""
        result = create_device_features(sample_data)
        assert 'device_unique_users' in result.columns
        d1_users = result[result['device_id'] == 'd1']['device_unique_users'].iloc[0]
        assert d1_users == 3


class TestEncoding:
    """Tests for categorical encoding."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with categorical features."""
        return pd.DataFrame({
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Firefox', 'Safari'],
            'sex': ['M', 'F', 'M'],
            'value': [100, 200, 300]
        })
    
    def test_one_hot_encoding(self, sample_data):
        """Test one-hot encoding creates correct columns."""
        result, info = encode_categorical_features(sample_data)
        
        # Check original columns are removed
        assert 'source' not in result.columns
        assert 'browser' not in result.columns
        assert 'sex' not in result.columns
        
        # Check encoded columns exist
        assert 'source_SEO' in result.columns or 'source_Ads' in result.columns
        assert 'browser_Chrome' in result.columns or 'browser_Firefox' in result.columns
    
    def test_encoding_preserves_numerical(self, sample_data):
        """Test that numerical columns are preserved."""
        result, _ = encode_categorical_features(sample_data)
        assert 'value' in result.columns


class TestScaling:
    """Tests for numerical scaling."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for scaling."""
        return pd.DataFrame({
            'purchase_value': [100.0, 200.0, 300.0, 400.0, 500.0],
            'age': [20, 30, 40, 50, 60]
        })
    
    def test_standard_scaling(self, sample_data):
        """Test standard scaling normalizes data."""
        result, scaler = scale_numerical_features(
            sample_data, 
            numerical_cols=['purchase_value', 'age'],
            scaler_type='standard'
        )
        
        # Standard scaled data should have mean ~0 and std ~1
        assert abs(result['purchase_value'].mean()) < 0.01
        assert abs(result['purchase_value'].std() - 1.0) < 0.2
    
    def test_minmax_scaling(self, sample_data):
        """Test min-max scaling bounds data."""
        result, scaler = scale_numerical_features(
            sample_data,
            numerical_cols=['purchase_value'],
            scaler_type='minmax'
        )
        
        # MinMax scaled data should be between 0 and 1
        assert result['purchase_value'].min() >= 0
        assert result['purchase_value'].max() <= 1


class TestPrepareFeatures:
    """Tests for feature preparation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for model preparation."""
        return pd.DataFrame({
            'user_id': [1, 2, 3],
            'signup_time': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'purchase_time': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-04']),
            'purchase_value': [100.0, 200.0, 300.0],
            'age': [25, 30, 35],
            'class': [0, 1, 0]
        })
    
    def test_separates_features_and_target(self, sample_data):
        """Test that features and target are properly separated."""
        X, y = prepare_features_for_modeling(sample_data)
        
        assert 'class' not in X.columns
        assert len(y) == len(sample_data)
        assert y.tolist() == [0, 1, 0]
    
    def test_drops_specified_columns(self, sample_data):
        """Test that specified columns are dropped."""
        X, y = prepare_features_for_modeling(sample_data)
        
        assert 'user_id' not in X.columns
        assert 'signup_time' not in X.columns
        assert 'purchase_time' not in X.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

