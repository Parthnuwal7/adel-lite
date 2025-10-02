#!/usr/bin/env python3

import sys
sys.path.append('.')

# Test function imports and calls
print("Testing function imports...")

try:
    from adel_lite.scoring_detailed import calculate_fk_score_detailed
    print("✓ calculate_fk_score_detailed imported successfully")
    print(f"Function: {calculate_fk_score_detailed}")
    print(f"Function.__name__: {calculate_fk_score_detailed.__name__}")
except Exception as e:
    print(f"✗ Error importing calculate_fk_score_detailed: {e}")

try:
    from adel_lite.fk_detection_detailed import calculate_fk_score
    print("✓ calculate_fk_score imported from fk_detection_detailed")
    print(f"Function: {calculate_fk_score}")
    print(f"Function.__name__: {calculate_fk_score.__name__}")
except Exception as e:
    print(f"✗ Error importing calculate_fk_score from fk_detection_detailed: {e}")

print("\nTesting a simple call...")
try:
    # Test parameters
    result = calculate_fk_score_detailed(
        coverage_fk_to_pk=0.8,
        coverage_pk_to_fk=0.9,
        pk_uniqueness=1.0,
        fk_null_ratio=0.0,
        datatype_match=1.0,
        pattern_match=1.0,
        name_fuzzy_similarity=0.8,
        constraint_violation_rate=0.0
    )
    print(f"✓ Direct call successful: {result}")
except Exception as e:
    print(f"✗ Error in direct call: {e}")
    import traceback
    traceback.print_exc()