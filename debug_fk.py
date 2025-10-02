"""
Debug version of simple demo to pinpoint FK detection error.
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_fk_detection():
    """Debug the FK detection error."""
    print("=== Debug FK Detection ===\n")
    
    # Create simple test data
    users_df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'name': ['alice', 'bob', 'charlie']
    })
    
    orders_df = pd.DataFrame({
        'order_id': [101, 102, 103],
        'user_id': [1, 2, 1]
    })
    
    print("Test data created successfully")
    
    try:
        from adel_lite.fk_detection_detailed import detect_foreign_keys_detailed
        
        print("Attempting FK detection...")
        
        # Monkey patch to get better error info
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        fk_relationships = detect_foreign_keys_detailed(
            orders_df, 'orders', users_df, 'users', ['user_id']
        )
        
        print(f"✓ FK detection completed successfully!")
        print(f"Found {len(fk_relationships)} relationships")
        
        for rel in fk_relationships:
            print(f"  - {rel['fk_table']}.{rel['fk_column']} → "
                  f"{rel['pk_table']}.{rel['pk_column']}: score={rel['score']:.3f}")
    
    except Exception as e:
        print(f"✗ Error in FK detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fk_detection()