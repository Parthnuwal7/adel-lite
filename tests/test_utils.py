"""
Tests for utility functions.
"""

import pytest
import pandas as pd
from adel_lite.utils import (
    infer_dtype, is_datetime, is_numeric, name_similarity,
    overlap_ratio, calculate_uniqueness_ratio, infer_semantic_type
)


class TestUtils:
    
    def test_infer_dtype(self):
        """Test data type inference."""
        assert infer_dtype(pd.Series([1, 2, 3])) == 'integer'
        assert infer_dtype(pd.Series([1.1, 2.2, 3.3])) == 'float'
        assert infer_dtype(pd.Series([True, False, True])) == 'boolean'
        assert infer_dtype(pd.Series(['a', 'b', 'c'])) == 'string'
        assert infer_dtype(pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02']))) == 'datetime'
    
    def test_is_datetime(self):
        """Test datetime detection."""
        datetime_series = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02']))
        string_series = pd.Series(['a', 'b', 'c'])
        
        assert is_datetime(datetime_series) == True
        assert is_datetime(string_series) == False
    
    def test_is_numeric(self):
        """Test numeric detection."""
        numeric_series = pd.Series([1, 2, 3])
        string_series = pd.Series(['a', 'b', 'c'])
        
        assert is_numeric(numeric_series) == True
        assert is_numeric(string_series) == False
    
    def test_name_similarity(self):
        """Test name similarity calculation."""
        assert name_similarity('customer_id', 'customer_id') == 1.0
        assert name_similarity('customer_id', 'cust_id') > 0.5
        assert name_similarity('customer_id', 'order_id') > 0.6
        assert name_similarity('abc', 'xyz') < 0.3
    
    def test_overlap_ratio(self):
        """Test overlap ratio calculation."""
        s1 = pd.Series([1, 2, 3, 4])
        s2 = pd.Series([1, 2, 5, 6])
        
        assert overlap_ratio(s1, s2) == 0.5  # 2 out of 4 overlap
        
        # Test with no overlap
        s3 = pd.Series([7, 8, 9])
        assert overlap_ratio(s1, s3) == 0.0
        
        # Test with complete overlap
        assert overlap_ratio(s1, s1) == 1.0
    
    def test_calculate_uniqueness_ratio(self):
        """Test uniqueness ratio calculation."""
        # All unique
        s1 = pd.Series([1, 2, 3, 4])
        assert calculate_uniqueness_ratio(s1) == 1.0
        
        # Some duplicates
        s2 = pd.Series([1, 1, 2, 3])
        assert calculate_uniqueness_ratio(s2) == 0.75  # 3 unique out of 4
        
        # All same
        s3 = pd.Series([1, 1, 1, 1])
        assert calculate_uniqueness_ratio(s3) == 0.25  # 1 unique out of 4
    
    def test_infer_semantic_type(self):
        """Test semantic type inference."""
        # Test ID column
        id_series = pd.Series([1, 2, 3, 4])
        semantic_type, subtype = infer_semantic_type(id_series, 'customer_id')
        assert semantic_type == 'id'
        assert subtype == 'primary'
        
        # Test datetime column
        dt_series = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02']))
        semantic_type, subtype = infer_semantic_type(dt_series, 'created_at')
        assert semantic_type == 'datetime'
        
        # Test categorical column
        cat_series = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
        semantic_type, subtype = infer_semantic_type(cat_series, 'category')
        assert semantic_type == 'categorical'
