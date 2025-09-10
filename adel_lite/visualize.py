"""
Schema visualization functionality using Graphviz.
"""

import graphviz
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def visualize(
    schema_output: Dict[str, Any], 
    relationships_output: Dict[str, Any], 
    filename: str = "schema_graph",
    format: str = "png",
    view: bool = False
) -> str:
    """
    Render schema graph with nodes as tables and edges as relationships.
    
    Args:
        schema_output: Output from schema() function
        relationships_output: Output from map_relationships() function
        filename: Output filename without extension
        format: Output format ('png', 'svg', 'pdf')
        view: Whether to open the rendered graph
        
    Returns:
        Path to the generated file
        
    Example:
        >>> from adel_lite import visualize
        >>> path = visualize(schema_output, relationships_output, 'my_schema', 'svg')
        >>> print(f"Schema saved to: {path}")
    """
    if format not in ['png', 'svg', 'pdf']:
        raise ValueError("Format must be one of: 'png', 'svg', 'pdf'")
    
    # Create directed graph
    dot = graphviz.Digraph(
        comment="Database Schema",
        format=format,
        graph_attr={
            'rankdir': 'TB',
            'bgcolor': 'white',
            'fontname': 'Arial',
            'fontsize': '12'
        },
        node_attr={
            'shape': 'record',
            'style': 'filled',
            'fillcolor': 'lightblue',
            'fontname': 'Arial',
            'fontsize': '10'
        },
        edge_attr={
            'fontname': 'Arial',
            'fontsize': '9',
            'color': 'darkblue'
        }
    )
    
    # Add table nodes with column information
    for table_name, table_info in schema_output.get('schemas', {}).items():
        columns = table_info.get('columns', [])
        
        # Build HTML-like table structure
        label_parts = [f"<table border='0' cellborder='1' cellspacing='0'>"]
        label_parts.append(f"<tr><td bgcolor='darkblue'><font color='white'><b>{table_name}</b></font></td></tr>")
        
        for col in columns[:10]:  # Limit to first 10 columns for readability
            col_name = col['name']
            col_type = col['dtype']
            nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
            
            label_parts.append(f"<tr><td align='left'>{col_name} ({col_type}) {nullable}</td></tr>")
        
        if len(columns) > 10:
            label_parts.append(f"<tr><td>... and {len(columns) - 10} more columns</td></tr>")
        
        label_parts.append("</table>")
        
        dot.node(table_name, label="<" + "".join(label_parts) + ">")
    
    # Add edges for foreign key relationships
    fks = relationships_output.get('foreign_keys', [])
    
    for fk in fks:
        src_table = fk['foreign_table']
        dst_table = fk['referenced_table']
        src_col = fk['foreign_column']
        dst_col = fk['referenced_column']
        confidence = fk['confidence']
        
        edge_label = f"{src_col} → {dst_col}\\nConf: {confidence:.2f}"
        
        # Different colors based on confidence
        if confidence >= 0.9:
            color = 'darkgreen'
        elif confidence >= 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        dot.edge(src_table, dst_table, label=edge_label, color=color)
    
    # Render and save
    try:
        output_path = dot.render(filename, cleanup=True)
        logger.info(f"Schema graph saved to: {output_path}")
        
        if view:
            dot.view()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to render graph: {str(e)}")
        raise


def create_simplified_graph(relationships_output: Dict[str, Any]) -> str:
    """
    Create a simplified graph showing only table relationships.
    
    Args:
        relationships_output: Output from map_relationships()
        
    Returns:
        Path to generated simplified graph
    """
    dot = graphviz.Digraph(
        'simple_schema',
        format='png',
        graph_attr={'rankdir': 'LR'},
        node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgray'}
    )
    
    # Extract unique tables from relationships
    tables = set()
    for fk in relationships_output.get('foreign_keys', []):
        tables.add(fk['foreign_table'])
        tables.add(fk['referenced_table'])
    
    # Add table nodes
    for table in tables:
        dot.node(table)
    
    # Add relationship edges
    for fk in relationships_output.get('foreign_keys', []):
        dot.edge(fk['foreign_table'], fk['referenced_table'])
    
    output_path = dot.render('simple_schema', cleanup=True)
    logger.info(f"Simplified schema graph saved to: {output_path}")
    
    return output_path
