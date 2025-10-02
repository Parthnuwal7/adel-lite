"""
Test the specific analyze_column_quality function.
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_analyze_column_quality():
    """Test the analyze_column_quality function directly."""
    print("=== Test analyze_column_quality ===\n")
    
    # Create simple test data
    test_series = pd.Series([1, 2, 3, 1, 2], name='user_id')
    
    try:
        from adel_lite.pk_detection_detailed import analyze_column_quality
        
        print("Testing analyze_column_quality...")
        result = analyze_column_quality(test_series, 'user_id')
        
        print(f"✓ analyze_column_quality completed successfully!")
        print(f"Pattern type: {result.get('pattern_type', 'unknown')}")
        print(f"Uniqueness: {result.get('uniqueness_ratio', 0):.3f}")
        
    except Exception as e:
        print(f"✗ Error in analyze_column_quality: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyze_column_quality()