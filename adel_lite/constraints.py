"""
Constraint detection functionality for intra-row relationships.
"""

import pandas as pd
from typing import List, Dict, Any
import logging
from .utils import validate_dataframes

logger = logging.getLogger(__name__)


def detect_constraints(
    df_list: List[pd.DataFrame], 
    table_names: List[str] = None, 
    threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Detect intra-row hard constraints in numeric/datetime columns:
    - GT: A > B (greater than)
    - EQ: A + B = C (equality/sum)
    
    Constraints must hold for >= threshold fraction of rows.
    """
    validate_dataframes(df_list)
    
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(df_list))]
    
    if len(table_names) != len(df_list):
        raise ValueError("Length of table_names must match length of df_list")
    
    constraints_output = {}
    
    for df, table_name in zip(df_list, table_names):
        logger.info(f"Detecting constraints for table: {table_name}")
        
        # Separate numeric and datetime columns to avoid invalid comparisons
        numeric_cols = [
            col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        datetime_cols = [
            col for col in df.columns 
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        
        detected_constraints = []
        
        # GT constraints within numeric columns only
        for col_a in numeric_cols:
            for col_b in numeric_cols:
                if col_a == col_b:
                    continue
                    
                valid_rows = df[[col_a, col_b]].dropna()
                if len(valid_rows) == 0:
                    continue
                    
                try:
                    gt_ratio = (valid_rows[col_a] > valid_rows[col_b]).mean()
                    if gt_ratio >= threshold:
                        detected_constraints.append({
                            'type': 'GT',
                            'colA': col_a,
                            'colB': col_b,
                            'confidence': round(gt_ratio, 4),
                            'description': f"{col_a} > {col_b}"
                        })
                except Exception as e:
                    logger.warning(f"Could not compare {col_a} and {col_b}: {str(e)}")
        
        # GT constraints within datetime columns only
        for col_a in datetime_cols:
            for col_b in datetime_cols:
                if col_a == col_b:
                    continue
                    
                valid_rows = df[[col_a, col_b]].dropna()
                if len(valid_rows) == 0:
                    continue
                    
                try:
                    gt_ratio = (valid_rows[col_a] > valid_rows[col_b]).mean()
                    if gt_ratio >= threshold:
                        detected_constraints.append({
                            'type': 'GT',
                            'colA': col_a,
                            'colB': col_b,
                            'confidence': round(gt_ratio, 4),
                            'description': f"{col_a} > {col_b}"
                        })
                except Exception as e:
                    logger.warning(f"Could not compare {col_a} and {col_b}: {str(e)}")
        
        # EQ constraints: A + B = C (numeric only)
        for col_a in numeric_cols:
            for col_b in numeric_cols:
                for col_c in numeric_cols:
                    if len({col_a, col_b, col_c}) != 3:  # All different
                        continue
                        
                    valid_rows = df[[col_a, col_b, col_c]].dropna()
                    if len(valid_rows) == 0:
                        continue
                    
                    try:
                        # Check if A + B = C
                        sum_vals = valid_rows[col_a] + valid_rows[col_b]
                        eq_ratio = (abs(sum_vals - valid_rows[col_c]) < 1e-10).mean()
                        
                        if eq_ratio >= threshold:
                            detected_constraints.append({
                                'type': 'EQ',
                                'colA': col_a,
                                'colB': col_b,
                                'colC': col_c,
                                'confidence': round(eq_ratio, 4),
                                'description': f"{col_a} + {col_b} = {col_c}"
                            })
                    except Exception as e:
                        logger.warning(f"Could not check equality constraint for {col_a}, {col_b}, {col_c}: {str(e)}")
        
        constraints_output[table_name] = detected_constraints
        logger.info(f"Found {len(detected_constraints)} constraints for {table_name}")
    
    return {
        'constraints': constraints_output,
        'table_count': len(df_list),
        'generated_at': pd.Timestamp.now().isoformat()
    }
