"""
Tests for data profiling functionality.
"""

import pytest
import pandas as pd
from adel_lite.profile import profile


class TestProfile:
    
    def test_basic_profiling(self, sample_df_list, sample_table_names):
        """Test basic profiling functionality."""
        result = profile(sample_df_list, sample_table_names)
        
        assert 'profiles' in result
        assert 'table_count' in result
        assert result['table_count'] == 2
    
    def test_column_statistics(self, sample_customers_df):
        """Test column statistics calculation."""
        result = profile([sample_customers_df], ['customers'])
        
        customers_profile = result['profiles']['customers']
        columns = customers_profile['columns']
        
        # Test customer_id column
        customer_id_col = next(col for col in columns if col['column_name'] == 'customer_id')
        assert customer_id_col['unique_count'] == 5
        assert customer_id_col['null_count'] == 0
        assert customer_id_col['uniqueness_ratio'] == 1.0
        assert customer_id_col['is_pk_candidate'] == True
    
    def test_semantic_type_detection(self, sample_customers_df):
        """Test semantic type detection."""
        result = profile([sample_customers_df], ['customers'])
        
        columns = result['profiles']['customers']['columns']
        
        # Check ID column
        customer_id_col = next(col for col in columns if col['column_name'] == 'customer_id')
        assert customer_id_col['semantic_type'] == 'id'
        assert customer_id_col['subtype'] == 'primary'
        
        # Check boolean column
        is_active_col = next(col for col in columns if col['column_name'] == 'is_active')
        assert is_active_col['semantic_type'] == 'boolean'
    
    def test_value_statistics(self, sample_customers_df):
        """Test value statistics calculation."""
        result = profile([sample_customers_df], ['customers'])
        
        columns = result['profiles']['customers']['columns']
        age_col = next(col for col in columns if col['column_name'] == 'age')
        
        value_stats = age_col['value_stats']
        assert 'min' in value_stats
        assert 'max' in value_stats
        assert 'mean' in value_stats
        assert value_stats['min'] == 25
        assert value_stats['max'] == 35
    
    def test_null_handling(self, df_with_nulls):
        """Test handling of null values."""
        result = profile([df_with_nulls], ['test'])
        
        columns = result['profiles']['test']['columns']
        value_col = next(col for col in columns if col['column_name'] == 'value')
        
        assert value_col['null_count'] == 1
        assert value_col['null_percentage'] == 25.0
