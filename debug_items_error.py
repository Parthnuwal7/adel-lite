"""
Targeted debug for FK detection error.
"""

import pandas as pd
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_specific_error():
    """Debug the specific FK detection error."""
    
    # Simple test data
    users_df = pd.DataFrame({'user_id': [1, 2, 3]})
    orders_df = pd.DataFrame({'user_id': [1, 2, 1]})
    
    print("Testing individual functions...")
    
    # Test the pattern detection that might be causing issues
    try:
        from adel_lite.pk_detection_detailed import detect_pattern_type
        
        values = [1, 2, 3]
        result = detect_pattern_type(values)
        print(f"detect_pattern_type result: {result}")
        print(f"Type: {type(result)}")
        
        if isinstance(result, tuple) and len(result) >= 3:
            pattern, confidence, metadata = result
            print(f"Pattern: {pattern}, Confidence: {confidence}")
            print(f"Metadata type: {type(metadata)}")
            if hasattr(metadata, 'items'):
                print("Metadata has items() method - this is good")
            else:
                print("ERROR: Metadata does not have items() method!")
    
    except Exception as e:
        print(f"Error in detect_pattern_type: {e}")
        traceback.print_exc()
    
    # Test the column analysis
    try:
        from adel_lite.pk_detection_detailed import analyze_column_quality
        
        series = users_df['user_id']
        result = analyze_column_quality(series, 'user_id')
        print(f"\nanalyze_column_quality completed successfully")
        print(f"Pattern metadata in result: {result.get('pattern_metadata', 'NOT FOUND')}")
        
    except Exception as e:
        print(f"Error in analyze_column_quality: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_specific_error()