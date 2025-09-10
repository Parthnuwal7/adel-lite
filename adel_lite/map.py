"""
Relationship mapping functionality for detecting PK/FK relationships.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from itertools import combinations
from .utils import (
    validate_dataframes, 
    overlap_ratio, 
    name_similarity,
    calculate_uniqueness_ratio,
    is_datetime,
    is_numeric
)

logger = logging.getLogger(__name__)


def map_relationships(
    df_list: List[pd.DataFrame], 
    table_names: List[str] = None,
    fk_threshold: float = 0.8,
    name_similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Detect relationships between DataFrames using heuristic algorithms.
    
    Args:
        df_list: List of pandas DataFrames
        table_names: Optional list of table names
        fk_threshold: Minimum coverage ratio for FK detection
        name_similarity_threshold: Minimum name similarity for relationship scoring
        
    Returns:
        Dictionary containing detected relationships
        
    Example:
        >>> import pandas as pd
        >>> from adel_lite import map_relationships
        >>> customers = pd.DataFrame({'customer_id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        >>> orders = pd.DataFrame({'order_id': [1, 2], 'customer_id': [1, 2]})
        >>> result = map_relationships([customers, orders], ['customers', 'orders'])
        >>> print(result)
    """
    validate_dataframes(df_list)
    
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(df_list))]
    
    if len(table_names) != len(df_list):
        raise ValueError("Length of table_names must match length of df_list")
    
    # Step 1: Detect Primary Keys
    primary_keys = _detect_primary_keys(df_list, table_names)
    
    # Step 2: Detect Composite Keys
    composite_keys = _detect_composite_keys(df_list, table_names)
    
    # Step 3: Detect Foreign Keys
    foreign_keys = _detect_foreign_keys(
        df_list, table_names, primary_keys, 
        fk_threshold, name_similarity_threshold
    )
    
    return {
        'primary_keys': primary_keys,
        'composite_keys': composite_keys,
        'foreign_keys': foreign_keys,
        'table_count': len(df_list),
        'generated_at': pd.Timestamp.now().isoformat()
    }


def _detect_primary_keys(df_list: List[pd.DataFrame], table_names: List[str]) -> List[Dict[str, Any]]:
    """
    Detect primary key candidates in each DataFrame.
    
    Args:
        df_list: List of DataFrames
        table_names: List of table names
        
    Returns:
        List of primary key candidates
    """
    primary_keys = []
    
    for df, table_name in zip(df_list, table_names):
        logger.info(f"Detecting primary keys for table: {table_name}")
        
        for col_name in df.columns:
            series = df[col_name]
            uniqueness_ratio = calculate_uniqueness_ratio(series)
            
            # PK criteria: uniqueness ratio ~1, not datetime/float, no nulls
            if (uniqueness_ratio == 1.0 and 
                not is_datetime(series) and 
                series.dtype != 'float64' and
                not series.isnull().any()):
                
                pk_info = {
                    'table': table_name,
                    'column': col_name,
                    'confidence': 1.0,
                    'uniqueness_ratio': uniqueness_ratio,
                    'null_count': series.isnull().sum()
                }
                primary_keys.append(pk_info)
                logger.info(f"Found PK candidate: {table_name}.{col_name}")
    
    return primary_keys


def _detect_composite_keys(df_list: List[pd.DataFrame], table_names: List[str]) -> List[Dict[str, Any]]:
    """
    Detect composite key candidates (2-column combinations).
    
    Args:
        df_list: List of DataFrames
        table_names: List of table names
        
    Returns:
        List of composite key candidates
    """
    composite_keys = []
    
    for df, table_name in zip(df_list, table_names):
        logger.info(f"Detecting composite keys for table: {table_name}")
        
        # Only check combinations of 2 columns for performance
        for col1, col2 in combinations(df.columns, 2):
            combined_series = df[col1].astype(str) + "_" + df[col2].astype(str)
            uniqueness_ratio = calculate_uniqueness_ratio(combined_series)
            
            if uniqueness_ratio == 1.0:
                ck_info = {
                    'table': table_name,
                    'columns': [col1, col2],
                    'confidence': uniqueness_ratio,
                    'uniqueness_ratio': uniqueness_ratio
                }
                composite_keys.append(ck_info)
                logger.info(f"Found CK candidate: {table_name}.({col1}, {col2})")
    
    return composite_keys


def _detect_foreign_keys(
    df_list: List[pd.DataFrame], 
    table_names: List[str], 
    primary_keys: List[Dict[str, Any]],
    fk_threshold: float,
    name_similarity_threshold: float
) -> List[Dict[str, Any]]:
    """
    Detect foreign key relationships between tables.
    
    Args:
        df_list: List of DataFrames
        table_names: List of table names
        primary_keys: Detected primary keys
        fk_threshold: Minimum overlap ratio for FK detection
        name_similarity_threshold: Minimum name similarity threshold
        
    Returns:
        List of foreign key relationships
    """
    foreign_keys = []
    df_dict = dict(zip(table_names, df_list))
    
    # Create PK lookup
    pk_dict = {}
    for pk in primary_keys:
        pk_dict[(pk['table'], pk['column'])] = pk
    
    logger.info("Detecting foreign key relationships")
    
    # Check each column against all potential PKs
    for fk_table, fk_df in df_dict.items():
        for fk_col in fk_df.columns:
            fk_series = fk_df[fk_col]
            
            # Check against all PKs in other tables
            for (pk_table, pk_col), pk_info in pk_dict.items():
                if pk_table == fk_table:  # Skip same table
                    continue
                
                pk_df = df_dict[pk_table]
                pk_series = pk_df[pk_col]
                
                # Calculate overlap ratio (subset check with tolerance)
                coverage_ratio = overlap_ratio(fk_series, pk_series)
                
                if coverage_ratio >= fk_threshold:
                    # Calculate name similarity for confidence scoring
                    name_sim = name_similarity(fk_col, pk_col)
                    
                    # Calculate overall confidence
                    confidence = (coverage_ratio + name_sim) / 2
                    
                    fk_info = {
                        'foreign_table': fk_table,
                        'foreign_column': fk_col,
                        'referenced_table': pk_table,
                        'referenced_column': pk_col,
                        'coverage_ratio': round(coverage_ratio, 4),
                        'name_similarity': round(name_sim, 4),
                        'confidence': round(confidence, 4)
                    }
                    
                    foreign_keys.append(fk_info)
                    logger.info(f"Found FK: {fk_table}.{fk_col} -> {pk_table}.{pk_col} (confidence: {confidence:.3f})")
    
    # Sort by confidence (highest first)
    foreign_keys.sort(key=lambda x: x['confidence'], reverse=True)
    
    return foreign_keys
