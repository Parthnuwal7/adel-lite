"""
Data profiling functionality for generating statistics and insights.
"""

import pandas as pd
from typing import List, Dict, Any
import logging
from .utils import (
    validate_dataframes, 
    infer_semantic_type, 
    calculate_uniqueness_ratio,
    infer_dtype,
    is_surrogate_key,
    calculate_entropy,
    detect_pattern_type,
    calculate_null_ratio
)
from .map import map_relationships

logger = logging.getLogger(__name__)


def profile(df_list: List[pd.DataFrame], table_names: List[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive profiling statistics for DataFrames.
    
    Args:
        df_list: List of pandas DataFrames
        table_names: Optional list of table names
        
    Returns:
        Dictionary containing profiling information for each table and column
        
    Example:
        >>> import pandas as pd
        >>> from adel_lite import profile
        >>> df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        >>> result = profile([df], ['users'])
        >>> print(result)
    """
    validate_dataframes(df_list)
    
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(df_list))]
    
    if len(table_names) != len(df_list):
        raise ValueError("Length of table_names must match length of df_list")
    
    # Detect relationships first if multiple tables
    relationships = {}
    if len(df_list) > 1:
        try:
            logger.info("Detecting relationships for profiling context")
            relationship_results = map_relationships(df_list, table_names)
            relationships = {
                'primary_keys': relationship_results.get('primary_keys', []),
                'foreign_keys': relationship_results.get('foreign_keys', []),
                'composite_keys': relationship_results.get('composite_keys', [])
            }
        except Exception as e:
            logger.warning(f"Relationship detection failed during profiling: {e}")
            relationships = {'error': str(e)}

    profiles = {}
    
    for df, table_name in zip(df_list, table_names):
        logger.info(f"Profiling table: {table_name}")
        
        column_profiles = []
        
        for col_name in df.columns:
            series = df[col_name]
            
            # Basic statistics
            total_count = len(series)
            null_count = series.isnull().sum()
            unique_count = series.nunique()
            uniqueness_ratio = calculate_uniqueness_ratio(series)
            null_ratio = calculate_null_ratio(series)
            
            # Enhanced analysis
            semantic_type, subtype = infer_semantic_type(series, col_name)
            entropy = calculate_entropy(series)
            pattern_type = detect_pattern_type(series, col_name)
            is_surrogate = is_surrogate_key(series, col_name)
            
            # Check relationship context
            detected_as_pk = any(pk['table'] == table_name and pk['column'] == col_name 
                               for pk in relationships.get('primary_keys', []))
            
            fk_relationships = [fk for fk in relationships.get('foreign_keys', [])
                              if fk['foreign_table'] == table_name and fk['foreign_column'] == col_name]
            
            # Enhanced PK candidate detection with confidence
            pk_confidence = 0.0
            if detected_as_pk:
                pk_info = next((pk for pk in relationships.get('primary_keys', [])
                              if pk['table'] == table_name and pk['column'] == col_name), {})
                pk_confidence = pk_info.get('confidence', 0.0)
            else:
                # Calculate potential PK confidence even if not detected
                if uniqueness_ratio >= 0.95 and null_count == 0:
                    pk_confidence = uniqueness_ratio
                    if is_surrogate:
                        pk_confidence = min(1.0, pk_confidence + 0.1)
            
            is_pk_candidate = (pk_confidence >= 0.95)
            
            # Value statistics
            value_stats = _calculate_value_stats(series)
            
            # Enhanced data quality metrics
            quality_metrics = {
                'completeness': round((total_count - null_count) / total_count, 4) if total_count > 0 else 0,
                'uniqueness': round(uniqueness_ratio, 4),
                'entropy': round(entropy, 4),
                'pattern_consistency': _calculate_pattern_consistency(series),
                'outlier_percentage': _calculate_outlier_percentage(series)
            }
            
            col_profile = {
                'column_name': col_name,
                'dtype': infer_dtype(series),
                'pandas_dtype': str(series.dtype),
                'semantic_type': semantic_type,
                'subtype': subtype,
                'pattern_type': pattern_type,
                'total_count': total_count,
                'null_count': null_count,
                'unique_count': unique_count,
                'uniqueness_ratio': round(uniqueness_ratio, 4),
                'null_percentage': round((null_count / total_count * 100), 2) if total_count > 0 else 0,
                'entropy': round(entropy, 4),
                'is_pk_candidate': is_pk_candidate,
                'pk_confidence': round(pk_confidence, 4),
                'is_surrogate_key': is_surrogate,
                'detected_as_pk': detected_as_pk,
                'foreign_key_relationships': fk_relationships,
                'quality_metrics': quality_metrics,
                'value_stats': value_stats
            }
            
            column_profiles.append(col_profile)
        
        # Enhanced table-level statistics
        data_quality_score = _calculate_table_quality_score(column_profiles)
        relationship_summary = _calculate_table_relationship_summary(
            table_name, relationships.get('primary_keys', []), relationships.get('foreign_keys', [])
        )
        
        table_profile = {
            'table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': column_profiles,
            'memory_usage_bytes': df.memory_usage(deep=True).sum(),
            'data_quality_score': data_quality_score,
            'relationship_summary': relationship_summary,
            'pk_candidates': [
                {
                    'column_name': col['column_name'],
                    'confidence': col['pk_confidence']
                } for col in column_profiles 
                if col['is_pk_candidate']
            ],
            'surrogate_keys': [
                col['column_name'] for col in column_profiles 
                if col['is_surrogate_key']
            ]
        }
        
        profiles[table_name] = table_profile
    
    # Enhanced global summary
    global_summary = _calculate_global_summary(profiles, relationships)
    
    return {
        'profiles': profiles,
        'relationships': relationships,
        'global_summary': global_summary,
        'table_count': len(df_list),
        'generated_at': pd.Timestamp.now().isoformat()
    }


def _calculate_value_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate detailed value statistics for a series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Dictionary of value statistics
    """
    stats = {}
    
    # Non-null series for calculations
    non_null_series = series.dropna()
    
    if len(non_null_series) == 0:
        return {'all_null': True}
    
    # Numeric statistics
    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            'min': float(non_null_series.min()),
            'max': float(non_null_series.max()),
            'mean': float(non_null_series.mean()),
            'median': float(non_null_series.median()),
            'std': float(non_null_series.std()) if len(non_null_series) > 1 else 0.0
        })
    
    # String statistics
    elif series.dtype == 'object':
        str_lengths = non_null_series.astype(str).str.len()
        stats.update({
            'min_length': int(str_lengths.min()),
            'max_length': int(str_lengths.max()),
            'avg_length': round(float(str_lengths.mean()), 2)
        })
    
    # Common statistics for all types
    value_counts = non_null_series.value_counts()
    stats.update({
        'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        'cardinality': len(value_counts)
    })
    
    return stats
