"""
Enhanced foreign key detection with advanced scoring and constraint validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
import logging
from .config import Config
from .scoring_detailed import calculate_fk_score_detailed as calculate_fk_score
from .pk_detection_detailed import analyze_column_quality, detect_pattern_type as detect_pattern_detailed

logger = logging.getLogger(__name__)


def calculate_bidirectional_coverage(fk_values: List[Any], pk_values: List[Any]) -> Dict[str, float]:
    """
    Calculate bidirectional coverage ratios between FK and PK columns.
    
    Args:
        fk_values: Values from foreign key column
        pk_values: Values from primary key column
        
    Returns:
        Dictionary with coverage metrics
    """
    # Remove nulls and convert to sets for faster operations
    fk_set = set(v for v in fk_values if v is not None)
    pk_set = set(v for v in pk_values if v is not None)
    
    if not fk_set or not pk_set:
        return {
            'fk_to_pk_coverage': 0.0,
            'pk_to_fk_coverage': 0.0,
            'intersection_size': 0,
            'fk_unique_count': len(fk_set),
            'pk_unique_count': len(pk_set),
            'coverage_quality': 'no_data'
        }
    
    # Calculate intersection
    intersection = fk_set & pk_set
    intersection_size = len(intersection)
    
    # Calculate coverage ratios
    fk_to_pk_coverage = intersection_size / len(fk_set)
    pk_to_fk_coverage = intersection_size / len(pk_set)
    
    # Determine coverage quality
    if fk_to_pk_coverage >= 0.95:
        if pk_to_fk_coverage >= 0.8:
            quality = 'excellent'
        elif pk_to_fk_coverage >= 0.5:
            quality = 'good'
        else:
            quality = 'fair'
    elif fk_to_pk_coverage >= 0.8:
        quality = 'fair'
    else:
        quality = 'poor'
    
    return {
        'fk_to_pk_coverage': fk_to_pk_coverage,
        'pk_to_fk_coverage': pk_to_fk_coverage,
        'intersection_size': intersection_size,
        'fk_unique_count': len(fk_set),
        'pk_unique_count': len(pk_set),
        'coverage_quality': quality
    }


def analyze_constraint_violations(fk_values: List[Any], pk_values: List[Any]) -> Dict[str, Any]:
    """
    Analyze referential integrity constraint violations.
    
    Args:
        fk_values: Values from foreign key column
        pk_values: Values from primary key column
        
    Returns:
        Analysis of constraint violations
    """
    # Remove nulls (nulls are allowed in FK)
    non_null_fk = [v for v in fk_values if v is not None]
    pk_set = set(v for v in pk_values if v is not None)
    
    if not non_null_fk or not pk_set:
        return {
            'violation_count': 0,
            'violation_ratio': 0.0,
            'violation_values': [],
            'total_non_null_fk': len(non_null_fk),
            'has_violations': False
        }
    
    # Find violations (FK values not in PK)
    violations = []
    for value in non_null_fk:
        if value not in pk_set:
            violations.append(value)
    
    violation_count = len(violations)
    violation_ratio = violation_count / len(non_null_fk)
    
    # Sample violations for inspection (limit to avoid huge lists)
    sample_violations = list(set(violations))[:10]
    
    return {
        'violation_count': violation_count,
        'violation_ratio': violation_ratio,
        'violation_values': sample_violations,
        'total_non_null_fk': len(non_null_fk),
        'has_violations': violation_count > 0
    }


def analyze_datatype_compatibility(fk_series: pd.Series, pk_series: pd.Series) -> Dict[str, Any]:
    """
    Analyze data type compatibility between FK and PK columns.
    
    Args:
        fk_series: Foreign key column series
        pk_series: Primary key column series
        
    Returns:
        Data type compatibility analysis
    """
    fk_dtype = str(fk_series.dtype)
    pk_dtype = str(pk_series.dtype)
    
    # Check if both are numeric
    fk_numeric = pd.api.types.is_numeric_dtype(fk_series)
    pk_numeric = pd.api.types.is_numeric_dtype(pk_series)
    
    # Check if both are string/object
    fk_string = pd.api.types.is_string_dtype(fk_series) or fk_dtype == 'object'
    pk_string = pd.api.types.is_string_dtype(pk_series) or pk_dtype == 'object'
    
    # Check if both are datetime
    fk_datetime = pd.api.types.is_datetime64_any_dtype(fk_series)
    pk_datetime = pd.api.types.is_datetime64_any_dtype(pk_series)
    
    # Determine compatibility
    if fk_dtype == pk_dtype:
        compatibility = 'exact_match'
        compatibility_score = 1.0
    elif (fk_numeric and pk_numeric) or (fk_string and pk_string) or (fk_datetime and pk_datetime):
        compatibility = 'compatible'
        compatibility_score = 0.8
    elif (fk_numeric and pk_string) or (fk_string and pk_numeric):
        # Could be string representations of numbers
        compatibility = 'convertible'
        compatibility_score = 0.6
    else:
        compatibility = 'incompatible'
        compatibility_score = 0.0
    
    return {
        'fk_dtype': fk_dtype,
        'pk_dtype': pk_dtype,
        'compatibility': compatibility,
        'compatibility_score': compatibility_score,
        'fk_numeric': fk_numeric,
        'pk_numeric': pk_numeric,
        'fk_string': fk_string,
        'pk_string': pk_string,
        'fk_datetime': fk_datetime,
        'pk_datetime': pk_datetime
    }


def analyze_pattern_consistency(fk_values: List[Any], pk_values: List[Any]) -> Dict[str, Any]:
    """
    Analyze pattern consistency between FK and PK columns.
    
    Args:
        fk_values: Foreign key values
        pk_values: Primary key values
        
    Returns:
        Pattern consistency analysis
    """
    # Detect patterns for both columns
    fk_pattern, fk_confidence, fk_metadata = detect_pattern_detailed(fk_values)
    pk_pattern, pk_confidence, pk_metadata = detect_pattern_detailed(pk_values)
    
    # Determine pattern consistency
    pattern_match = fk_pattern == pk_pattern
    pattern_consistency_score = 0.0
    
    if pattern_match and fk_pattern != 'unknown':
        # Both have the same identifiable pattern
        pattern_consistency_score = min(fk_confidence, pk_confidence)
    elif fk_pattern == 'unknown' and pk_pattern == 'unknown':
        # Neither has a clear pattern - neutral
        pattern_consistency_score = 0.5
    elif fk_pattern == 'unknown' or pk_pattern == 'unknown':
        # One has pattern, one doesn't - partial penalty
        pattern_consistency_score = 0.3
    else:
        # Different patterns - penalty
        pattern_consistency_score = 0.1
    
    return {
        'fk_pattern': fk_pattern,
        'fk_pattern_confidence': fk_confidence,
        'pk_pattern': pk_pattern,
        'pk_pattern_confidence': pk_confidence,
        'pattern_match': pattern_match,
        'pattern_consistency_score': pattern_consistency_score,
        'fk_pattern_metadata': fk_metadata,
        'pk_pattern_metadata': pk_metadata
    }


def calculate_name_similarity_detailed(fk_column: str, pk_column: str, pk_table: str) -> Dict[str, float]:
    """
    Calculate detailed name similarity metrics.
    
    Args:
        fk_column: Foreign key column name
        pk_column: Primary key column name
        pk_table: Primary key table name
        
    Returns:
        Dictionary with similarity metrics
    """
    from .utils import name_similarity
    
    # Direct column name similarity
    direct_similarity = name_similarity(fk_column, pk_column)
    
    # Check for common FK naming patterns
    fk_lower = fk_column.lower()
    pk_lower = pk_column.lower()
    pk_table_lower = pk_table.lower()
    
    # Pattern: table_id -> table.id
    table_id_pattern = name_similarity(fk_lower, f"{pk_table_lower}_{pk_lower}")
    table_id_pattern2 = name_similarity(fk_lower, f"{pk_table_lower}{pk_lower}")
    
    # Pattern: tableid -> table.id
    concatenated_pattern = name_similarity(fk_lower.replace('_', ''), f"{pk_table_lower}{pk_lower}")
    
    # Pattern: id_table -> table.id
    id_table_pattern = name_similarity(fk_lower, f"{pk_lower}_{pk_table_lower}")
    
    # Best similarity score
    all_similarities = [
        direct_similarity,
        table_id_pattern,
        table_id_pattern2,
        concatenated_pattern,
        id_table_pattern
    ]
    
    best_similarity = max(all_similarities)
    
    return {
        'direct_similarity': direct_similarity,
        'table_id_pattern': table_id_pattern,
        'concatenated_pattern': concatenated_pattern,
        'id_table_pattern': id_table_pattern,
        'best_similarity': best_similarity,
        'naming_convention_detected': best_similarity > direct_similarity
    }


def detect_foreign_keys_detailed(fk_df: pd.DataFrame, fk_table: str, 
                                pk_df: pd.DataFrame, pk_table: str,
                                pk_candidates: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Detect foreign key relationships with detailed analysis and scoring.
    
    Args:
        fk_df: DataFrame containing potential foreign key columns
        fk_table: Name of the FK table
        pk_df: DataFrame containing potential primary key columns
        pk_table: Name of the PK table
        pk_candidates: List of PK column names to check (if None, check all columns)
        
    Returns:
        List of foreign key relationships with detailed metadata
    """
    if fk_df.empty or pk_df.empty:
        logger.warning(f"Empty DataFrame(s) for FK detection: {fk_table} -> {pk_table}")
        return []
    
    if pk_candidates is None:
        pk_candidates = list(pk_df.columns)
    
    fk_relationships = []
    
    logger.info(f"Analyzing FK relationships: {fk_table} ({len(fk_df.columns)} cols) -> "
                f"{pk_table} ({len(pk_candidates)} PK candidates)")
    
    for fk_column in fk_df.columns:
        for pk_column in pk_candidates:
            try:
                fk_series = fk_df[fk_column]
                pk_series = pk_df[pk_column]
                
                # Skip if either column is entirely null
                if fk_series.isnull().all() or pk_series.isnull().all():
                    continue
                
                # Extract values for analysis
                fk_values = fk_series.tolist()
                pk_values = pk_series.tolist()
                
                # Bidirectional coverage analysis
                coverage_analysis = calculate_bidirectional_coverage(fk_values, pk_values)
                
                # Skip if no intersection at all
                if coverage_analysis['intersection_size'] == 0:
                    continue
                
                # Constraint violation analysis
                constraint_analysis = analyze_constraint_violations(fk_values, pk_values)
                
                # Data type compatibility
                dtype_analysis = analyze_datatype_compatibility(fk_series, pk_series)
                
                # Pattern consistency
                pattern_analysis = analyze_pattern_consistency(fk_values, pk_values)
                
                # Name similarity
                name_similarity_analysis = calculate_name_similarity_detailed(fk_column, pk_column, pk_table)
                
                # Analyze FK column quality
                fk_column_analysis = analyze_column_quality(fk_series, fk_column)
                
                # Analyze PK column quality  
                pk_column_analysis = analyze_column_quality(pk_series, pk_column)
                
                # Calculate comprehensive FK score
                fk_features = {
                    'fk_to_pk_coverage': coverage_analysis['fk_to_pk_coverage'],
                    'pk_to_fk_coverage': coverage_analysis['pk_to_fk_coverage'],
                    'constraint_violation_ratio': constraint_analysis['violation_ratio'],
                    'dtype_compatibility_score': dtype_analysis['compatibility_score'],
                    'pattern_consistency_score': pattern_analysis['pattern_consistency_score'],
                    'name_similarity': name_similarity_analysis['best_similarity'],
                    'fk_null_ratio': fk_column_analysis['null_ratio'],
                    'fk_uniqueness_ratio': fk_column_analysis['uniqueness_ratio'],
                    'coverage_quality': coverage_analysis['coverage_quality']
                }
                
                try:
                    logger.info(f"About to call calculate_fk_score with parameters:")
                    logger.info(f"  coverage_fk_to_pk: {fk_features['fk_to_pk_coverage']}")
                    logger.info(f"  coverage_pk_to_fk: {fk_features['pk_to_fk_coverage']}")
                    logger.info(f"  pk_uniqueness: {pk_column_analysis['uniqueness_ratio']}")
                    
                    scoring_result = calculate_fk_score(
                        coverage_fk_to_pk=fk_features['fk_to_pk_coverage'],
                        coverage_pk_to_fk=fk_features['pk_to_fk_coverage'],
                        pk_uniqueness=pk_column_analysis['uniqueness_ratio'],
                        fk_null_ratio=fk_features['fk_null_ratio'],
                        datatype_match=fk_features['dtype_compatibility_score'],
                        pattern_match=fk_features['pattern_consistency_score'],
                        name_fuzzy_similarity=fk_features['name_similarity'],
                        constraint_violation_rate=fk_features['constraint_violation_ratio']
                    )
                    logger.info(f"Successfully calculated FK score: {scoring_result['score']}")
                except Exception as e:
                    logger.error(f"Error in calculate_fk_score: {e}")
                    logger.error(f"fk_features type: {type(fk_features)}")
                    logger.error(f"fk_features: {fk_features}")
                    raise
                
                # Create relationship record
                relationship = {
                    'fk_table': fk_table,
                    'fk_column': fk_column,
                    'pk_table': pk_table,
                    'pk_column': pk_column,
                    'score': scoring_result['score'],
                    'decision': scoring_result['decision'],
                    'explanation': scoring_result['explanation'],
                    'detailed_explanation': scoring_result['detailed_explanation'],
                    'features': fk_features,
                    'coverage_analysis': coverage_analysis,
                    'constraint_analysis': constraint_analysis,
                    'dtype_analysis': dtype_analysis,
                    'pattern_analysis': pattern_analysis,
                    'name_similarity_analysis': name_similarity_analysis,
                    'fk_column_analysis': fk_column_analysis,
                    'metadata': {
                        'fk_cardinality': fk_column_analysis['unique_count'],
                        'pk_cardinality': len(set(v for v in pk_values if v is not None)),
                        'intersection_size': coverage_analysis['intersection_size'],
                        'fk_total_rows': len(fk_values),
                        'pk_total_rows': len(pk_values),
                        'relationship_type': _determine_relationship_type(
                            fk_column_analysis['uniqueness_ratio'],
                            coverage_analysis['pk_to_fk_coverage']
                        )
                    }
                }
                
                fk_relationships.append(relationship)
                
                # Log significant relationships
                if scoring_result['score'] > 0.5:
                    logger.info(f"FK relationship {fk_table}.{fk_column} -> {pk_table}.{pk_column}: "
                              f"score={scoring_result['score']:.3f}, "
                              f"decision={scoring_result['decision']}, "
                              f"coverage={coverage_analysis['fk_to_pk_coverage']:.2f}")
            
            except Exception as e:
                logger.error(f"Error analyzing FK relationship {fk_table}.{fk_column} -> "
                           f"{pk_table}.{pk_column}: {e}")
                continue
    
    # Sort by score (highest first)
    fk_relationships.sort(key=lambda x: x['score'], reverse=True)
    
    # Log summary
    accepted_relationships = [r for r in fk_relationships if r['decision'] == 'accepted']
    logger.info(f"Found {len(accepted_relationships)} accepted FK relationships: "
                f"{fk_table} -> {pk_table}")
    
    return fk_relationships


def _determine_relationship_type(fk_uniqueness: float, pk_coverage: float) -> str:
    """
    Determine the type of relationship (one-to-one, one-to-many, etc.).
    
    Args:
        fk_uniqueness: Uniqueness ratio of FK column
        pk_coverage: Coverage of PK by FK (how much of PK is referenced)
        
    Returns:
        Relationship type string
    """
    if fk_uniqueness >= 0.95:
        if pk_coverage >= 0.95:
            return 'one-to-one'
        else:
            return 'one-to-one-partial'
    else:
        if pk_coverage >= 0.8:
            return 'one-to-many-high-coverage'
        elif pk_coverage >= 0.3:
            return 'one-to-many-medium-coverage'
        else:
            return 'one-to-many-low-coverage'


def filter_best_fk_relationships(relationships: List[Dict[str, Any]], 
                                max_per_fk_column: int = 1) -> List[Dict[str, Any]]:
    """
    Filter to keep only the best FK relationships, avoiding duplicates.
    
    Args:
        relationships: List of FK relationships
        max_per_fk_column: Maximum relationships per FK column
        
    Returns:
        Filtered list of best relationships
    """
    if not relationships:
        return []
    
    # Group by FK column
    fk_column_groups = {}
    for rel in relationships:
        fk_key = f"{rel['fk_table']}.{rel['fk_column']}"
        if fk_key not in fk_column_groups:
            fk_column_groups[fk_key] = []
        fk_column_groups[fk_key].append(rel)
    
    # Keep best relationships for each FK column
    filtered_relationships = []
    for fk_key, group in fk_column_groups.items():
        # Sort by score and take top N
        group.sort(key=lambda x: x['score'], reverse=True)
        filtered_relationships.extend(group[:max_per_fk_column])
    
    # Sort final list by score
    filtered_relationships.sort(key=lambda x: x['score'], reverse=True)
    
    return filtered_relationships


def compare_fk_relationships(relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple FK relationships and provide analysis.
    
    Args:
        relationships: List of FK relationships to compare
        
    Returns:
        Comparison analysis
    """
    if not relationships:
        return {'comparison': 'No relationships to compare'}
    
    # Basic statistics
    scores = [r['score'] for r in relationships]
    decisions = [r['decision'] for r in relationships]
    
    comparison = {
        'total_relationships': len(relationships),
        'accepted_count': decisions.count('accepted'),
        'rejected_count': decisions.count('rejected'),
        'ambiguous_count': decisions.count('ambiguous'),
        'score_statistics': {
            'max': max(scores),
            'min': min(scores),
            'avg': sum(scores) / len(scores),
            'median': sorted(scores)[len(scores) // 2]
        }
    }
    
    # Coverage distribution
    coverages = [r['coverage_analysis']['fk_to_pk_coverage'] for r in relationships]
    comparison['coverage_statistics'] = {
        'max': max(coverages),
        'min': min(coverages),
        'avg': sum(coverages) / len(coverages)
    }
    
    # Relationship type distribution
    rel_types = [r['metadata']['relationship_type'] for r in relationships]
    type_counts = {}
    for rel_type in rel_types:
        type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
    comparison['relationship_type_distribution'] = type_counts
    
    # Quality indicators
    high_quality = [r for r in relationships if r['score'] > 0.8]
    medium_quality = [r for r in relationships if 0.5 < r['score'] <= 0.8]
    low_quality = [r for r in relationships if r['score'] <= 0.5]
    
    comparison['quality_distribution'] = {
        'high_quality': len(high_quality),
        'medium_quality': len(medium_quality),
        'low_quality': len(low_quality)
    }
    
    # Best relationship
    if relationships:
        best = max(relationships, key=lambda x: x['score'])
        comparison['best_relationship'] = {
            'fk_table': best['fk_table'],
            'fk_column': best['fk_column'],
            'pk_table': best['pk_table'],
            'pk_column': best['pk_column'],
            'score': best['score'],
            'relationship_type': best['metadata']['relationship_type']
        }
    
    return comparison