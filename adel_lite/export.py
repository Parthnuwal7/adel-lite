"""
Schema export functionality for multiple formats.
"""

import json
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def export_schema(
    schema_output: Dict[str, Any], 
    format: str = "json",
    output_file: str = None
) -> str:
    """
    Export schema into different formats.
    
    Args:
        schema_output: Schema output from build_meta() or other functions
        format: Export format ('json', 'yaml', 'avro', 'ddl')
        output_file: Optional output file path
        
    Returns:
        Exported schema as string
        
    Example:
        >>> from adel_lite import export_schema
        >>> json_content = export_schema(meta_output, 'json')
        >>> yaml_content = export_schema(meta_output, 'yaml')
        >>> ddl_content = export_schema(meta_output, 'ddl')
    """
    format = format.lower()
    
    if format == "json":
        content = json.dumps(schema_output, indent=2, default=str)
    elif format == "yaml":
        content = yaml.dump(schema_output, default_flow_style=False, sort_keys=False)
    elif format == "avro":
        content = _export_avro(schema_output)
    elif format == "ddl":
        content = _export_ddl(schema_output)
    else:
        raise ValueError(f"Unsupported export format: {format}. Supported: json, yaml, avro, ddl")
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Schema exported to: {output_file}")
    
    return content


def _export_avro(schema_output: Dict[str, Any]) -> str:
    """
    Generate simplified Avro schema from schema output.
    """
    avro_schemas = []
    
    for table in schema_output.get('tables', []):
        table_name = table['name']
        fields = []
        
        for field in table.get('fields', []):
            avro_field = {
                "name": field['name'],
                "type": _map_dtype_to_avro(field['dtype']),
                "doc": f"Semantic type: {field.get('semantic_type', 'unknown')}"
            }
            
            # Handle nullable fields
            if field.get('nullable', True):
                avro_field['type'] = ["null", avro_field['type']]
                avro_field['default'] = None
            
            fields.append(avro_field)
        
        table_schema = {
            "type": "record",
            "name": table_name,
            "namespace": "adel_lite.generated",
            "doc": f"Generated schema for table {table_name}",
            "fields": fields
        }
        
        avro_schemas.append(table_schema)
    
    return json.dumps(avro_schemas, indent=2)


def _map_dtype_to_avro(dtype: str) -> str:
    """Map high-level dtype to Avro types."""
    dtype_mapping = {
        'integer': 'int',
        'int': 'int',
        'float': 'float',
        'double': 'double', 
        'boolean': 'boolean',
        'string': 'string',
        'text': 'string',
        'categorical': 'string',
        'datetime': 'string'  # Avro doesn't have native datetime
    }
    return dtype_mapping.get(dtype.lower(), 'string')


def _export_ddl(schema_output: Dict[str, Any]) -> str:
    """Generate SQL DDL (CREATE TABLE) statements."""
    ddl_statements = []
    
    # Generate CREATE TABLE statements
    for table in schema_output.get('tables', []):
        table_name = table['name']
        fields = table.get('fields', [])
        primary_key = table.get('primary_key')
        
        column_definitions = []
        
        for field in fields:
            col_name = field['name']
            sql_type = _map_dtype_to_sql(field['dtype'])
            nullable = '' if field.get('nullable', True) else 'NOT NULL'
            
            column_def = f"    {col_name} {sql_type} {nullable}".strip()
            column_definitions.append(column_def)
        
        # Add primary key constraint
        if primary_key:
            column_definitions.append(f"    PRIMARY KEY ({primary_key})")
        
        create_table = f"CREATE TABLE {table_name} (\n"
        create_table += ",\n".join(column_definitions)
        create_table += "\n);"
        
        ddl_statements.append(create_table)
    
    # Generate ALTER TABLE statements for foreign keys
    for relationship in schema_output.get('relationships', []):
        if relationship['type'] == 'FK':
            fk_table = relationship['table']
            fk_column = relationship['column']
            ref_table = relationship['references']['table']
            ref_column = relationship['references']['column']
            
            alter_statement = (
                f"ALTER TABLE {fk_table} "
                f"ADD FOREIGN KEY ({fk_column}) "
                f"REFERENCES {ref_table}({ref_column});"
            )
            ddl_statements.append(alter_statement)
    
    return "\n\n".join(ddl_statements)


def _map_dtype_to_sql(dtype: str) -> str:
    """Map high-level dtype to SQL data types."""
    sql_mapping = {
        'integer': 'INTEGER',
        'int': 'INTEGER',
        'float': 'FLOAT',
        'double': 'DOUBLE PRECISION',
        'boolean': 'BOOLEAN',
        'datetime': 'TIMESTAMP',
        'string': 'VARCHAR(255)',
        'text': 'TEXT',
        'categorical': 'VARCHAR(100)'
    }
    return sql_mapping.get(dtype.lower(), 'TEXT')
