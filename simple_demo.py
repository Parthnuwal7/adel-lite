"""
Simple demonstration of enhanced adel-lite functionality.
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_demo():
    """Simple demo without full package imports."""
    print("=== Simple Enhanced Adel-Lite Demo ===\n")
    
    # Create sample data
    print("1. Creating sample datasets...")
    
    users_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'username': ['alice', 'bob', 'charlie', 'diana', 'eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                 'diana@example.com', 'eve@example.com']
    })
    
    orders_df = pd.DataFrame({
        'order_id': ['ORD-001', 'ORD-002', 'ORD-003', 'ORD-004'],
        'user_id': [1, 2, 1, 3],  # References users.user_id
        'amount': [99.99, 149.50, 75.00, 200.00]
    })
    
    print(f"✓ Created users table: {len(users_df)} rows, {len(users_df.columns)} columns")
    print(f"✓ Created orders table: {len(orders_df)} rows, {len(orders_df.columns)} columns")
    
    # Import and test configuration
    print("\n2. Testing Configuration System...")
    try:
        from adel_lite.config import Config
        print(f"✓ Configuration loaded successfully")
        print(f"  - PK uniqueness threshold: {Config.pk_uniqueness_threshold}")
        print(f"  - FK coverage threshold: {Config.fk_coverage_threshold}")
        print(f"  - Name similarity threshold: {Config.name_similarity_threshold}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
    # Test enhanced PK detection
    print("\n3. Testing Enhanced Primary Key Detection...")
    try:
        from adel_lite.pk_detection_detailed import detect_primary_keys_detailed
        
        pk_candidates = detect_primary_keys_detailed(users_df, 'users')
        print(f"✓ Found {len(pk_candidates)} PK candidates in users table:")
        
        for candidate in pk_candidates[:3]:
            print(f"  - {candidate['column']}: score={candidate['score']:.3f}, "
                  f"decision={candidate['decision']}")
    except Exception as e:
        print(f"✗ Error in PK detection: {e}")
    
    # Test enhanced profiling
    print("\n4. Testing Enhanced Profiling...")
    try:
        from adel_lite.profile_detailed import profile_detailed
        
        profile = profile_detailed(users_df, 'users')
        print(f"✓ Profile generated for users table:")
        print(f"  - Columns analyzed: {profile['successful_profiles']}/{profile['column_count']}")
        print(f"  - Memory usage: {profile['table_statistics']['memory_usage_mb']:.2f} MB")
        
        # Show some column insights
        for col_name, col_profile in profile['column_profiles'].items():
            if 'insights' in col_profile and col_profile['insights']:
                print(f"  - {col_name}: {col_profile['insights'][0]}")
    except Exception as e:
        print(f"✗ Error in profiling: {e}")
    
    # Test enhanced FK detection
    print("\n5. Testing Enhanced Foreign Key Detection...")
    try:
        from adel_lite.fk_detection_detailed import detect_foreign_keys_detailed
        
        fk_relationships = detect_foreign_keys_detailed(
            orders_df, 'orders', users_df, 'users', ['user_id']
        )
        
        print(f"✓ Found {len(fk_relationships)} potential FK relationships:")
        for rel in fk_relationships[:3]:
            print(f"  - {rel['fk_table']}.{rel['fk_column']} → "
                  f"{rel['pk_table']}.{rel['pk_column']}: score={rel['score']:.3f}")
    except Exception as e:
        print(f"✗ Error in FK detection: {e}")
    
    print("\n=== Demo Complete ===")
    print("\nThe enhanced framework successfully provides:")
    print("✓ Configurable detection thresholds")
    print("✓ Weighted scoring with detailed explanations")
    print("✓ Comprehensive column profiling and statistics")
    print("✓ Advanced foreign key relationship detection")
    print("✓ Pattern recognition and data quality analysis")

if __name__ == "__main__":
    simple_demo()