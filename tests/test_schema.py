"""
Tests for schema generation functionality.
"""

import pytest
import pandas as pd
from adel_lite.schema import schema


class TestSchema:
    
    def test_basic_schema_generation(self, sample_df_list, sample_table_names):
        """Test basic schema generation."""
        result = schema(sample_df_list, sample_table_names)
        
        assert 'schemas' in result
        assert 'table_count' in result
        assert 'generated_at' in result
        assert result['table_count'] == 2
        assert 'customers' in result['schemas']
        assert 'orders' in result['schemas']
    
    def test_schema_without_table_names(self, sample_df_list):
        """Test schema generation without providing table names."""
        result = schema(sample_df_list)
        
        assert 'table_0' in result['schemas']
        assert 'table_1' in result['schemas']
    
    def test_column_details(self, sample_customers_df):
        """Test that column details are correctly captured."""
        result = schema([sample_customers_df], ['customers'])
        
        customers_schema = result['schemas']['customers']
        columns = customers_schema['columns']
        
        # Find customer_id column
        customer_id_col = next(col for col in columns if col['name'] == 'customer_id')
        assert customer_id_col['dtype'] == 'integer'
        assert customer_id_col['nullable'] == False
        
        # Find name column  
        name_col = next(col for col in columns if col['name'] == 'name')
        assert name_col['dtype'] == 'string'
    
    def test_empty_dataframe(self, empty_df):
        """Test schema generation with empty DataFrame."""
        result = schema([empty_df], ['empty'])
        
        assert result['schemas']['empty']['row_count'] == 0
        assert result['schemas']['empty']['column_count'] == 0
    
    def test_invalid_input(self):
        """Test schema generation with invalid input."""
        with pytest.raises(ValueError):
            schema([])  # Empty list
        
        with pytest.raises(ValueError):
            schema(['not_a_dataframe'])  # Invalid type
    
    def test_mismatched_table_names(self, sample_df_list):
        """Test error when table names don't match DataFrame count."""
        with pytest.raises(ValueError):
            schema(sample_df_list, ['only_one_name'])  # Too few names
