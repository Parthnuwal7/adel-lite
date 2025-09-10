"""
Meta structure builder for combining all analysis results.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def build_meta(
    schema_output: Dict[str, Any],
    profile_output: Dict[str, Any], 
    relationships_output: Dict[str, Any],
    constraints_output: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build final meta.json structure combining all analysis results.
    
    Args:
        schema_output: Output from schema() function
        profile_output: Output from profile() function  
        relationships_output: Output from map_relationships() function
        constraints_output: Optional output from detect_constraints() function
        
    Returns:
        Complete meta structure as dictionary
        
    Example:
        >>> from adel_lite import build_meta
        >>> meta = build_meta(schema_out, profile_out, relationships_out, constraints_out)
        >>> print(json.dumps(meta, indent=2))
    """
    if constraints_output is None:
        constraints_output = {'constraints': {}}
    
    meta = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'generator': 'adel-lite',
            'version': '0.1.0'
        },
        'tables': [],
        'relationships': []
    }
    
    # Build tables section
    for table_name, table_schema in schema_output.get('schemas', {}).items():
        logger.info(f"Building meta for table: {table_name}")
        
        # Get corresponding profile data
        table_profile = profile_output.get('profiles', {}).get(table_name, {})
        table_constraints = constraints_output.get('constraints', {}).get(table_name, [])
        
        # Build fields list
        fields = []
        primary_key = None
        
        profile_columns = table_profile.get('columns', [])
        
        for col_schema in table_schema.get('columns', []):
            col_name = col_schema['name']
            
            # Find matching profile data
            col_profile = next(
                (c for c in profile_columns if c['column_name'] == col_name), 
                {}
            )
            
            # Determine primary key
            if col_profile.get('subtype') == 'primary':
                primary_key = col_name
            
            field_info = {
                'name': col_name,
                'dtype': col_schema['dtype'],
                'pandas_dtype': col_schema.get('pandas_dtype', ''),
                'semantic_type': col_profile.get('semantic_type', 'unknown'),
                'subtype': col_profile.get('subtype', ''),
                'nullable': col_schema.get('nullable', True),
                'position': col_schema.get('position', 0),
                'statistics': {
                    'unique_count': col_profile.get('unique_count', 0),
                    'null_count': col_profile.get('null_count', 0),
                    'uniqueness_ratio': col_profile.get('uniqueness_ratio', 0.0),
                    'null_percentage': col_profile.get('null_percentage', 0.0)
                }
            }
            
            # Add value statistics if available
            if 'value_stats' in col_profile:
                field_info['value_statistics'] = col_profile['value_stats']
            
            fields.append(field_info)
        
        # Build table metadata
        table_meta = {
            'name': table_name,
            'primary_key': primary_key,
            'row_count': table_schema.get('row_count', 0),
            'column_count': table_schema.get('column_count', 0),
            'fields': fields,
            'constraints': []
        }
        
        # Add constraints
        for constraint in table_constraints:
            constraint_info = {
                'type': constraint['type'],
                'description': constraint.get('description', ''),
                'confidence': constraint['confidence'],
                'columns': []
            }
            
            # Add involved columns
            if 'colA' in constraint:
                constraint_info['columns'].append(constraint['colA'])
            if 'colB' in constraint:
                constraint_info['columns'].append(constraint['colB'])
            if 'colC' in constraint:
                constraint_info['columns'].append(constraint['colC'])
            
            table_meta['constraints'].append(constraint_info)
        
        meta['tables'].append(table_meta)
    
    # Build relationships section
    for fk in relationships_output.get('foreign_keys', []):
        relationship = {
            'type': 'foreign_key',
            'foreign_table': fk['foreign_table'],
            'foreign_column': fk['foreign_column'],
            'referenced_table': fk['referenced_table'],  
            'referenced_column': fk['referenced_column'],
            'confidence': fk['confidence'],
            'coverage_ratio': fk.get('coverage_ratio', 0.0),
            'name_similarity': fk.get('name_similarity', 0.0)
        }
        meta['relationships'].append(relationship)
    
    # Add summary statistics
    meta['summary'] = {
        'total_tables': len(meta['tables']),
        'total_relationships': len(meta['relationships']),
        'total_constraints': sum(len(t['constraints']) for t in meta['tables']),
        'tables_with_pk': len([t for t in meta['tables'] if t['primary_key']]),
        'avg_columns_per_table': sum(t['column_count'] for t in meta['tables']) / len(meta['tables']) if meta['tables'] else 0
    }
    
    return meta


def format_meta_summary(meta: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the meta structure.
    
    Args:
        meta: Meta structure from build_meta()
        
    Returns:
        Formatted summary string
    """
    summary = meta.get('summary', {})
    
    report = f"""
Schema Analysis Summary
======================
Tables: {summary.get('total_tables', 0)}
Relationships: {summary.get('total_relationships', 0)}  
Constraints: {summary.get('total_constraints', 0)}
Tables with Primary Key: {summary.get('tables_with_pk', 0)}
Average Columns per Table: {summary.get('avg_columns_per_table', 0):.1f}

Table Details:
"""
    
    for table in meta.get('tables', []):
        pk_info = f" (PK: {table['primary_key']})" if table['primary_key'] else " (No PK)"
        constraint_count = len(table.get('constraints', []))
        
        report += f"  - {table['name']}: {table['column_count']} columns, {table['row_count']} rows{pk_info}"
        if constraint_count > 0:
            report += f", {constraint_count} constraints"
        report += "\n"
    
    if meta.get('relationships'):
        report += "\nRelationships:\n"
        for rel in meta['relationships']:
            report += f"  - {rel['foreign_table']}.{rel['foreign_column']} → {rel['referenced_table']}.{rel['referenced_column']} (confidence: {rel['confidence']:.2f})\n"
    
    return report
