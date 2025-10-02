"""
Utility functions for data type inference and analysis.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Tuple, Union
from fuzzywuzzy import fuzz
import logging

logger = logging.getLogger(__name__)


def infer_dtype(series: pd.Series) -> str:
    """
    Map pandas dtype to high-level type category.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        High-level data type string
        
    Example:
        >>> series = pd.Series([1, 2, 3])
        >>> infer_dtype(series)
        'integer'
    """
    dtype_str = str(series.dtype).lower()
    
    if 'int' in dtype_str:
        return 'integer'
    elif 'float' in dtype_str:
        return 'float'
    elif 'bool' in dtype_str:
        return 'boolean'
    elif 'datetime' in dtype_str or 'timestamp' in dtype_str:
        return 'datetime'
    elif 'object' in dtype_str or 'string' in dtype_str:
        return 'string'
    elif 'category' in dtype_str:
        return 'categorical'
    else:
        return 'unknown'


def is_datetime(series: pd.Series) -> bool:
    """
    Check if a series contains datetime-like values.
    
    Args:
        series: Pandas Series to check
        
    Returns:
        True if series contains datetime values
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    # Try to parse as datetime for string columns
    if series.dtype == 'object':
        try:
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False
            
            # Try common datetime formats first to avoid the warning
            common_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']
            
            for fmt in common_formats:
                try:
                    pd.to_datetime(sample, format=fmt, errors='raise')
                    return True
                except (ValueError, TypeError):
                    continue
            
            # Fall back to automatic parsing (REMOVE the deprecated argument)
            pd.to_datetime(sample, errors='raise')  # ✅ Removed infer_datetime_format=True
            return True
        except (ValueError, TypeError):
            return False
    
    return False



def is_numeric(series: pd.Series) -> bool:
    """
    Check if a series contains numeric values.
    
    Args:
        series: Pandas Series to check
        
    Returns:
        True if series contains numeric values
    """
    return pd.api.types.is_numeric_dtype(series)


def name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two column names using fuzzy matching.
    
    Args:
        name1: First column name
        name2: Second column name
        
    Returns:
        Similarity score between 0 and 1
        
    Example:
        >>> name_similarity("customer_id", "cust_id")
        0.8
    """
    if name1 == name2:
        return 1.0
    
    # Normalize names
    name1_clean = name1.lower().strip()
    name2_clean = name2.lower().strip()
    
    # Use fuzzy string matching
    ratio = fuzz.ratio(name1_clean, name2_clean) / 100.0
    
    return ratio


def overlap_ratio(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate overlap ratio between two series: |A ∩ B| / |A|.
    
    Args:
        series1: First series (A)
        series2: Second series (B)
        
    Returns:
        Overlap ratio between 0 and 1
        
    Example:
        >>> s1 = pd.Series([1, 2, 3, 4])
        >>> s2 = pd.Series([1, 2, 5, 6])
        >>> overlap_ratio(s1, s2)
        0.5
    """
    if len(series1) == 0:
        return 0.0
    
    set1 = set(series1.dropna().unique())
    set2 = set(series2.dropna().unique())
    
    if len(set1) == 0:
        return 0.0
    
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1)


def calculate_uniqueness_ratio(series: pd.Series) -> float:
    """
    Calculate uniqueness ratio: unique_count / total_count.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Uniqueness ratio between 0 and 1
    """
    total_count = len(series.dropna())
    if total_count == 0:
        return 0.0
    
    unique_count = series.nunique()
    return unique_count / total_count


def infer_semantic_type(series: pd.Series, column_name: str) -> Tuple[str, str]:
    """
    Infer semantic type and subtype for a column.
    
    Args:
        series: Pandas Series to analyze
        column_name: Name of the column
        
    Returns:
        Tuple of (semantic_type, subtype)
    """
    col_name_lower = column_name.lower()
    uniqueness_ratio = calculate_uniqueness_ratio(series)
    nunique = series.nunique(dropna=True)
    
    # Check for datetime FIRST (before ID detection)
    if is_datetime(series):
        return ('datetime', 'timestamp')
    
    # Check for ID columns
    if ('id' in col_name_lower or 
        col_name_lower.endswith('_key') or 
        uniqueness_ratio > 0.95):
        
        if uniqueness_ratio == 1.0:
            return ('id', 'primary')
        elif uniqueness_ratio > 0.95:
            return ('id', 'potential_primary')
        else:
            return ('id', 'foreign')
    
    # Check for boolean
    if series.dtype == 'bool':
        return ('boolean', 'flag')
    
    # Improved categorical detection
    if uniqueness_ratio < 0.1 and nunique < 50:
        return ('categorical', 'discrete')
    
    # Handle string vs categorical distinction
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        if nunique < 50:  # Low cardinality string columns are categorical
            return ('categorical', 'discrete')
        else:
            return ('text', 'string')
    
    # Check for numeric
    if is_numeric(series):
        return ('number', 'continuous')
    
    # Default to text
    return ('text', 'string')

def validate_dataframes(df_list: List[pd.DataFrame]) -> bool:
    """
    Validate input DataFrame list.
    
    Args:
        df_list: List of pandas DataFrames
        
    Returns:
        True if valid, raises ValueError if not
        
    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(df_list, list):
        raise ValueError("Input must be a list of pandas DataFrames")
    
    if len(df_list) == 0:
        raise ValueError("Input list cannot be empty")
    
    for i, df in enumerate(df_list):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Item at index {i} is not a pandas DataFrame")
        
        if df.empty:
            logger.warning(f"DataFrame at index {i} is empty")
    
    return True


def is_surrogate_key(series: pd.Series) -> bool:
    """
    Check if a series represents a surrogate key (auto-incrementing integer).
    
    Args:
        series: Pandas Series to check
        
    Returns:
        True if likely a surrogate key, False otherwise
    """
    try:
        # Must be numeric and unique
        if not pd.api.types.is_numeric_dtype(series) or series.nunique() != len(series.dropna()):
            return False
        
        clean_series = series.dropna().sort_values()
        if len(clean_series) < 2:
            return False
        
        # Check if it's sequential integers starting from 1 or 0
        diffs = clean_series.diff().dropna()
        return all(diffs == 1) and (clean_series.iloc[0] in [0, 1])
    
    except Exception:
        return False


def calculate_entropy(series: pd.Series) -> float:
    """
    Calculate Shannon entropy of a series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Shannon entropy value
    """
    try:
        import numpy as np
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    except Exception:
        return 0.0


def detect_pattern_type(series: pd.Series) -> str:
    """
    Detect the pattern type of a series (email, phone, UUID, etc.).
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Pattern type string
    """
    import re
    
    str_series = series.astype(str)
    
    # Email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if str_series.str.match(email_pattern).sum() / len(str_series) > 0.8:
        return 'email'
    
    # UUID pattern
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if str_series.str.match(uuid_pattern, case=False).sum() / len(str_series) > 0.8:
        return 'uuid'
    
    # Phone pattern (simple)
    phone_pattern = r'^\+?[\d\s\-\(\)]{7,15}$'
    if str_series.str.match(phone_pattern).sum() / len(str_series) > 0.8:
        return 'phone'
    
    return 'unknown'


def calculate_null_ratio(series: pd.Series) -> float:
    """
    Calculate the ratio of null values in a series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Ratio of null values (0.0 to 1.0)
    """
    return series.isnull().sum() / len(series) if len(series) > 0 else 0.0
