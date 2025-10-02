"""
Enhanced Adel-Lite Library - Complete Enhancement Framework

This module demonstrates the new enhanced detection capabilities with configurable
thresholds, weighted scoring, and comprehensive analysis.
"""

import pandas as pd
import sys
import os

# Import enhanced modules directly to avoid full package loading
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adel_lite.config import Config
from adel_lite.pk_detection_detailed import detect_primary_keys_detailed, get_best_primary_key_candidate
from adel_lite.fk_detection_detailed import detect_foreign_keys_detailed
from adel_lite.profile_detailed import profile_detailed
from adel_lite.map_relationships_detailed import map_relationships_detailed

def demo_enhanced_detection():
    """
    Demonstrate the enhanced detection capabilities.
    """
    print("=== Enhanced Adel-Lite Detection Demo ===\n")
    
    # Create sample data for demonstration
    print("Creating sample datasets...")
    
    # Users table (with strong PK)
    users_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'username': ['alice', 'bob', 'charlie', 'diana', 'eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                 'diana@example.com', 'eve@example.com'],
        'created_at': pd.date_range('2023-01-01', periods=5)
    })
    
    # Orders table (with FK to users)
    orders_df = pd.DataFrame({
        'order_id': ['ORD-001', 'ORD-002', 'ORD-003', 'ORD-004'],
        'user_id': [1, 2, 1, 3],  # References users.user_id
        'amount': [99.99, 149.50, 75.00, 200.00],
        'status': ['completed', 'pending', 'completed', 'shipped']
    })
    
    # Products table (standalone)
    products_df = pd.DataFrame({
        'product_id': ['PROD-A', 'PROD-B', 'PROD-C'],
        'name': ['Widget A', 'Widget B', 'Widget C'],
        'price': [29.99, 49.99, 19.99]
    })
    
    tables = {
        'users': users_df,
        'orders': orders_df,
        'products': products_df
    }
    
    print(f"Created {len(tables)} sample tables\n")
    
    # 1. Demonstrate Enhanced Primary Key Detection
    print("1. Enhanced Primary Key Detection")
    print("-" * 40)
    
    for table_name, df in tables.items():
        print(f"\nAnalyzing table: {table_name}")
        pk_candidates = detect_primary_keys_detailed(df, table_name)
        
        print(f"Found {len(pk_candidates)} PK candidates:")
        for candidate in pk_candidates[:3]:  # Show top 3
            print(f"  - {candidate['column']}: score={candidate['score']:.3f}, "
                  f"decision={candidate['decision']}")
            print(f"    Pattern: {candidate['analysis']['pattern_type']}, "
                  f"Uniqueness: {candidate['analysis']['uniqueness_ratio']:.1%}")
    
    # 2. Demonstrate Enhanced Profile Function
    print("\n\n2. Enhanced Table Profiling")
    print("-" * 40)
    
    profile = profile_detailed(users_df, 'users')
    print(f"\nProfile for 'users' table:")
    print(f"  - Total columns: {profile['column_count']}")
    print(f"  - Memory usage: {profile['table_statistics']['memory_usage_mb']:.2f} MB")
    print(f"  - Null ratio: {profile['table_statistics']['null_cell_ratio']:.1%}")
    
    print(f"\nColumn insights:")
    for col_name, col_profile in profile['column_profiles'].items():
        if 'insights' in col_profile:
            insights = col_profile['insights']
            if insights:
                print(f"  - {col_name}: {insights[0]}")
    
    # 3. Demonstrate Comprehensive Relationship Mapping
    print("\n\n3. Comprehensive Relationship Mapping")
    print("-" * 40)
    
    analysis = map_relationships_detailed(tables, output_format='summary')
    
    print(f"\nRelationship Analysis Summary:")
    summary = analysis['relationship_summary']
    print(f"  - Total tables: {summary['total_tables']}")
    print(f"  - Tables with strong PK: {summary['tables_with_strong_pk']}")
    print(f"  - Total relationships found: {summary['total_relationships']}")
    print(f"  - Strong relationships: {summary['relationship_strength_distribution']['strong']}")
    
    print(f"\nAccepted Relationships:")
    for rel in analysis['accepted_relationships']:
        print(f"  - {rel['fk_table']}.{rel['fk_column']} → "
              f"{rel['pk_table']}.{rel['pk_column']} (score: {rel['score']:.3f})")
    
    # 4. Demonstrate Configuration Flexibility
    print("\n\n4. Configuration Flexibility")
    print("-" * 40)
    
    print(f"Current Configuration:")
    print(f"  - PK uniqueness threshold: {Config.pk_uniqueness_threshold}")
    print(f"  - FK coverage threshold: {Config.fk_coverage_threshold}")
    print(f"  - Name similarity threshold: {Config.name_similarity_threshold}")
    
    # Show how to modify configuration
    custom_config = {
        'pk_uniqueness_threshold': 0.95,  # More lenient
        'fk_coverage_threshold': 0.7,     # More lenient
        'strict_mode': False
    }
    
    print(f"\nUsing custom configuration for stricter analysis...")
    strict_analysis = map_relationships_detailed(tables, 
                                               output_format='summary',
                                               config_overrides=custom_config)
    
    print(f"  - Relationships with custom config: "
          f"{strict_analysis['relationship_summary']['total_relationships']}")
    
    print("\n=== Demo Complete ===")
    print("\nThe enhanced framework provides:")
    print("✓ Configurable detection thresholds")
    print("✓ Weighted scoring with explanations")
    print("✓ Pattern recognition (UUID, email, etc.)")
    print("✓ Comprehensive profiling and statistics")
    print("✓ Network analysis of table relationships")
    print("✓ Detailed constraint violation checking")
    print("✓ Export capabilities (CSV, Markdown)")

if __name__ == "__main__":
    demo_enhanced_detection()