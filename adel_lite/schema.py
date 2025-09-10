"""
Schema generation functionality for DataFrames.
"""

import pandas as pd
from typing import List, Dict, Any
import logging
from .utils import infer_dtype, validate_dataframes

logger = logging.getLogger(__name__)


def schema(df_list: List[pd.DataFrame], table_names: List[str] = None) -> Dict[str, Any]:
    """
    Generate structural schema for a list of DataFrames.
    
    Args:
        df_list: List of pandas DataFrames
        table_names: Optional list of table names. If None, uses 'table_0', 'table_1', etc.
        
    Returns:
        Dictionary containing schema information for each table
        
    Example:
        >>> import pandas as pd
        >>> from adel_lite import schema
        >>> df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        >>> df2 = pd.DataFrame({'order_id': [1, 2], 'customer_id': [1, 2]})
        >>> result = schema([df1, df2], ['customers', 'orders'])
        >>> print(result)
    """
    validate_dataframes(df_list)
    
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(df_list))]
    
    if len(table_names) != len(df_list):
        raise ValueError("Length of table_names must match length of df_list")
    
    schemas = {}
    
    for i, (df, table_name) in enumerate(zip(df_list, table_names)):
        logger.info(f"Generating schema for table: {table_name}")
        
        columns = []
        for col_name in df.columns:
            col_info = {
                'name': col_name,
                'dtype': infer_dtype(df[col_name]),
                'pandas_dtype': str(df[col_name].dtype),
                'nullable': df[col_name].isnull().any(),
                'position': df.columns.get_loc(col_name)
            }
            columns.append(col_info)
        
        table_schema = {
            'table_name': table_name,
            'columns': columns,
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        schemas[table_name] = table_schema
    
    return {
        'schemas': schemas,
        'table_count': len(df_list),
        'generated_at': pd.Timestamp.now().isoformat()
    }
