"""
Tests for constraint detection functionality.
"""

import pytest
import pandas as pd
from adel_lite.constraints import detect_constraints


class TestConstraints:
    
    def test_gt_constraint_detection(self):
        """Test greater-than constraint detection."""
        df = pd.DataFrame({
            'start_date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'end_date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-04']),
            'min_value': [10, 20, 30],
            'max_value': [15, 25, 35]  # Add numeric constraint test
        })
        
        result = detect_constraints([df], ['test'], threshold=0.9)
        
        constraints = result['constraints']['test']
        
        # Should find end_date > start_date AND max_value > min_value
        gt_constraints = [c for c in constraints if c['type'] == 'GT']
        assert len(gt_constraints) >= 1
        
        # Check for datetime constraint
        date_constraint = next((c for c in gt_constraints 
                            if c['colA'] == 'end_date' and c['colB'] == 'start_date'), None)
        assert date_constraint is not None
        assert date_constraint['confidence'] == 1.0
    
    def test_eq_constraint_detection(self, sample_orders_df):
        """Test equality constraint detection."""
        result = detect_constraints([sample_orders_df], ['orders'], threshold=0.9)
        
        constraints = result['constraints']['orders']
        
        # Should find amount + tax = total
        eq_constraints = [c for c in constraints if c['type'] == 'EQ']
        assert len(eq_constraints) > 0
        
        sum_constraint = next((c for c in eq_constraints 
                             if set([c['colA'], c['colB']]) == {'amount', 'tax'} 
                             and c['colC'] == 'total'), None)
        assert sum_constraint is not None
        assert sum_constraint['confidence'] >= 0.9
    
    def test_no_constraints(self):
        """Test when no constraints exist."""
        df = pd.DataFrame({
            'random1': [1, 5, 3, 8],
            'random2': [2, 1, 9, 4],
            'random3': [7, 3, 2, 6]
        })
        
        result = detect_constraints([df], ['test'])
        
        constraints = result['constraints']['test']
        assert len(constraints) == 0
    
    def test_threshold_filtering(self):
        """Test constraint filtering by threshold."""
        # Create data where constraint holds for 2/3 of rows
        df = pd.DataFrame({
            'a': [1, 2, 5],
            'b': [2, 3, 1]  # a > b for 2/3 rows
        })
        
        # With high threshold, should find no constraints
        result_high = detect_constraints([df], ['test'], threshold=0.9)
        assert len(result_high['constraints']['test']) == 0
        
        # With low threshold, should find constraint
        result_low = detect_constraints([df], ['test'], threshold=0.6)
        gt_constraints = [c for c in result_low['constraints']['test'] if c['type'] == 'GT']
        assert len(gt_constraints) > 0
