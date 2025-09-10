"""
Data sampling functionality for quick data inspection.
"""

import pandas as pd
from typing import List, Dict, Any, Union
import logging
from .utils import validate_dataframes

logger = logging.getLogger(__name__)


def sample(
    df_list: List[pd.DataFrame], 
    table_names: List[str] = None,
    n: int = 5,
    method: str = 'head'
) -> Dict[str, Any]:
    """
    Return sample rows from each DataFrame for inspection.
    
    Args:
        df_list: List of pandas DataFrames
        table_names: Optional list of table names
        n: Number of rows to sample
        method: Sampling method ('head', 'tail', 'random')
        
    Returns:
        Dictionary containing sample data for each table
        
    Example:
        >>> import pandas as pd
        >>> from adel_lite import sample
        >>> df = pd.DataFrame({'id': range(100), 'value': range(100, 200)})
        >>> result = sample([df], ['data'], n=3, method='random')
        >>> print(result)
    """
    validate_dataframes(df_list)
    
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(df_list))]
    
    if len(table_names) != len(df_list):
        raise ValueError("Length of table_names must match length of df_list")
    
    if method not in ['head', 'tail', 'random']:
        raise ValueError("Method must be one of: 'head', 'tail', 'random'")
    
    samples = {}
    
    for df, table_name in zip(df_list, table_names):
        logger.info(f"Sampling {n} rows from table: {table_name} using method: {method}")
        
        # Get sample based on method
        if method == 'head':
            sample_df = df.head(n)
        elif method == 'tail':
            sample_df = df.tail(n)
        elif method == 'random':
            sample_size = min(n, len(df))
            sample_df = df.sample(n=sample_size, random_state=42) if len(df) > 0 else df
        
        # Convert to records for JSON serialization
        sample_records = sample_df.to_dict('records')
        
        table_sample = {
            'table_name': table_name,
            'method': method,
            'requested_rows': n,
            'actual_rows': len(sample_records),
            'total_rows': len(df),
            'columns': list(df.columns),
            'sample_data': sample_records
        }
        
        samples[table_name] = table_sample
    
    return {
        'samples': samples,
        'table_count': len(df_list),
        'sampling_method': method,
        'generated_at': pd.Timestamp.now().isoformat()
    }


def sample_by_condition(
    df_list: List[pd.DataFrame], 
    conditions: List[str],
    table_names: List[str] = None,
    n: int = 5
) -> Dict[str, Any]:
    """
    Sample rows based on specific conditions for each DataFrame.
    
    Args:
        df_list: List of pandas DataFrames
        conditions: List of query conditions (pandas query syntax)
        table_names: Optional list of table names
        n: Maximum number of rows to return per condition
        
    Returns:
        Dictionary containing conditional sample data
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'id': [1,2,3,4], 'value': [10,20,30,40]})
        >>> result = sample_by_condition([df], ['value > 20'], ['data'])
        >>> print(result)
    """
    validate_dataframes(df_list)
    
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(df_list))]
    
    if len(conditions) != len(df_list):
        raise ValueError("Length of conditions must match length of df_list")
    
    samples = {}
    
    for df, condition, table_name in zip(df_list, conditions, table_names):
        logger.info(f"Sampling from table: {table_name} with condition: {condition}")
        
        try:
            # Apply condition
            if condition.strip():
                filtered_df = df.query(condition)
            else:
                filtered_df = df
            
            # Sample from filtered data
            sample_size = min(n, len(filtered_df))
            if sample_size > 0:
                sample_df = filtered_df.head(sample_size)
            else:
                sample_df = pd.DataFrame(columns=df.columns)
            
            # Convert to records
            sample_records = sample_df.to_dict('records')
            
            table_sample = {
                'table_name': table_name,
                'condition': condition,
                'matching_rows': len(filtered_df),
                'sampled_rows': len(sample_records),
                'total_rows': len(df),
                'sample_data': sample_records
            }
            
            samples[table_name] = table_sample
            
        except Exception as e:
            logger.error(f"Error applying condition '{condition}' to table {table_name}: {str(e)}")
            samples[table_name] = {
                'table_name': table_name,
                'condition': condition,
                'error': str(e),
                'sample_data': []
            }
    
    return {
        'conditional_samples': samples,
        'table_count': len(df_list),
        'generated_at': pd.Timestamp.now().isoformat()
    }
