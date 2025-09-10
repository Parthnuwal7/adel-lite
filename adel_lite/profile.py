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
    infer_dtype
)

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
            
            # Semantic analysis
            semantic_type, subtype = infer_semantic_type(series, col_name)
            
            # Primary key candidate detection
            is_pk_candidate = (
                uniqueness_ratio == 1.0 and 
                null_count == 0 and 
                semantic_type == 'id'
            )
            
            # Value statistics
            value_stats = _calculate_value_stats(series)
            
            col_profile = {
                'column_name': col_name,
                'dtype': infer_dtype(series),
                'pandas_dtype': str(series.dtype),
                'semantic_type': semantic_type,
                'subtype': subtype,
                'total_count': total_count,
                'null_count': null_count,
                'unique_count': unique_count,
                'uniqueness_ratio': round(uniqueness_ratio, 4),
                'null_percentage': round((null_count / total_count * 100), 2) if total_count > 0 else 0,
                'is_pk_candidate': is_pk_candidate,
                'value_stats': value_stats
            }
            
            column_profiles.append(col_profile)
        
        # Table-level statistics
        table_profile = {
            'table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': column_profiles,
            'memory_usage_bytes': df.memory_usage(deep=True).sum(),
            'pk_candidates': [
                col['column_name'] for col in column_profiles 
                if col['is_pk_candidate']
            ]
        }
        
        profiles[table_name] = table_profile
    
    return {
        'profiles': profiles,
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
