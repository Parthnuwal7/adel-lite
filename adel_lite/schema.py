"""
Schema generation functionality for DataFrames.
"""

import pandas as pd
from typing import List, Dict, Any, Union
import logging
from .utils import infer_dtype, validate_dataframes

logger = logging.getLogger(__name__)


def schema(
    df_input: Union[pd.DataFrame, List[pd.DataFrame]], 
    table_names: Union[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Generate structural schema for a DataFrame or list of DataFrames.
    
    Args:
        df_input: Single pandas DataFrame OR list of pandas DataFrames
        table_names: Optional string (for single DF) or list of strings (for multiple DFs)
        
    Returns:
        Dictionary containing schema information for each table
        
    Examples:
        >>> import pandas as pd
        >>> from adel_lite import schema
        
        # Single DataFrame
        >>> df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        >>> result = schema(df, 'customers')
        
        # Multiple DataFrames
        >>> df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        >>> df2 = pd.DataFrame({'order_id': [1, 2], 'customer_id': [1, 2]})
        >>> result = schema([df1, df2], ['customers', 'orders'])
    """
    
    # Handle single DataFrame vs list of DataFrames
    if isinstance(df_input, pd.DataFrame):
        # Single DataFrame case
        df_list = [df_input]
        
        # Handle table_names for single DataFrame
        if table_names is None:
            table_names = ["table_0"]
        elif isinstance(table_names, str):
            table_names = [table_names]
        elif isinstance(table_names, list) and len(table_names) == 1:
            table_names = table_names  # Already a list with one item
        else:
            raise ValueError("For single DataFrame, table_names must be a string or single-item list")
            
    elif isinstance(df_input, list):
        # List of DataFrames case
        df_list = df_input
        
        # Handle table_names for list of DataFrames
        if table_names is None:
            table_names = [f"table_{i}" for i in range(len(df_list))]
        elif isinstance(table_names, list):
            if len(table_names) != len(df_list):
                raise ValueError("Length of table_names must match length of DataFrame list")
        else:
            raise ValueError("For multiple DataFrames, table_names must be a list")
    else:
        raise ValueError("df_input must be a pandas DataFrame or list of DataFrames")
    
    # Validate the DataFrames
    validate_dataframes(df_list)
    
    # Rest of the existing schema logic remains exactly the same
    schemas = {}
    
    for i, (df, table_name) in enumerate(zip(df_list, table_names)):
        logger.info(f"Generating schema for table: {table_name}")
        
        columns = []
        for col_name in df.columns:
            col_info = {
                'name': col_name,
                'dtype': infer_dtype(df[col_name]),
                'pandas_dtype': str(df[col_name].dtype),
                'nullable': bool(df[col_name].isnull().any()),  
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
