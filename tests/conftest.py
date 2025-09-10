"""
Pytest configuration and fixtures for adel-lite tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_customers_df():
    """Sample customers DataFrame for testing."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
        'age': [25, 30, 35, 28, 32],
        'created_at': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']),
        'is_active': [True, True, False, True, True]
    })


@pytest.fixture
def sample_orders_df():
    """Sample orders DataFrame for testing."""
    return pd.DataFrame({
        'order_id': [101, 102, 103, 104],
        'customer_id': [1, 2, 1, 3],
        'amount': [100.0, 150.0, 75.0, 200.0],
        'tax': [10.0, 15.0, 7.5, 20.0],
        'total': [110.0, 165.0, 82.5, 220.0],
        'order_date': pd.to_datetime(['2020-01-15', '2020-01-16', '2020-01-17', '2020-01-18']),
        'status': ['completed', 'pending', 'completed', 'cancelled']
    })


@pytest.fixture
def sample_df_list(sample_customers_df, sample_orders_df):
    """List of sample DataFrames."""
    return [sample_customers_df, sample_orders_df]


@pytest.fixture
def sample_table_names():
    """Sample table names."""
    return ['customers', 'orders']


@pytest.fixture
def empty_df():
    """Empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def single_column_df():
    """Single column DataFrame."""
    return pd.DataFrame({'id': [1, 2, 3]})


@pytest.fixture
def df_with_nulls():
    """DataFrame with null values."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, None, 30, 40],
        'text': ['a', 'b', None, 'd']
    })
