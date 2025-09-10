"""
Tests for relationship mapping functionality.
"""

import pytest
import pandas as pd
from adel_lite.map import map_relationships


class TestMapRelationships:
    
    def test_foreign_key_detection(self, sample_df_list, sample_table_names):
        """Test foreign key detection."""
        result = map_relationships(sample_df_list, sample_table_names)
        
        assert 'foreign_keys' in result
        assert 'primary_keys' in result
        
        # Should detect orders.customer_id -> customers.customer_id
        fks = result['foreign_keys']
        assert len(fks) > 0
        
        # Find the FK relationship
        fk = next((fk for fk in fks if fk['foreign_table'] == 'orders' and fk['foreign_column'] == 'customer_id'), None)
        assert fk is not None
        assert fk['referenced_table'] == 'customers'
        assert fk['referenced_column'] == 'customer_id'
    
    def test_primary_key_detection(self, sample_df_list, sample_table_names):
        """Test primary key detection."""
        result = map_relationships(sample_df_list, sample_table_names)
        
        pks = result['primary_keys']
        
        # Should detect customer_id and order_id as PKs
        pk_columns = [(pk['table'], pk['column']) for pk in pks]
        assert ('customers', 'customer_id') in pk_columns
        assert ('orders', 'order_id') in pk_columns
    
    def test_confidence_scoring(self, sample_df_list, sample_table_names):
        """Test confidence scoring for relationships."""
        result = map_relationships(sample_df_list, sample_table_names)
        
        fks = result['foreign_keys']
        for fk in fks:
            assert 0 <= fk['confidence'] <= 1
            assert 'coverage_ratio' in fk
            assert 'name_similarity' in fk
    
    def test_no_relationships(self):
        """Test when no relationships exist."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})
        
        result = map_relationships([df1, df2], ['table1', 'table2'])
        
        assert len(result['foreign_keys']) == 0
        assert len(result['primary_keys']) >= 0  # Might detect PKs anyway
    
    def test_composite_key_detection(self):
        """Test composite key detection."""
        df = pd.DataFrame({
            'col1': [1, 1, 2, 2],
            'col2': ['a', 'b', 'a', 'b'],
            'value': [10, 20, 30, 40]
        })
        
        result = map_relationships([df], ['test'])
        
        # Should detect (col1, col2) as composite key
        cks = result['composite_keys']
        assert len(cks) > 0
        
        ck = cks[0]
        assert set(ck['columns']) == {'col1', 'col2'}
