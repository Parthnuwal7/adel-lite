"""
Enhanced relationship mapping with comprehensive detection and scoring.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from .config import Config
from .pk_detection_detailed import detect_primary_keys_detailed, get_best_primary_key_candidate
from .fk_detection_detailed import detect_foreign_keys_detailed, filter_best_fk_relationships
from .profile_detailed import profile_detailed

logger = logging.getLogger(__name__)


def analyze_table_relationships(tables: Dict[str, pd.DataFrame], 
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze relationships between multiple tables.
    
    Args:
        tables: Dictionary mapping table names to DataFrames
        config: Optional configuration overrides
        
    Returns:
        Comprehensive relationship analysis
    """
    if not tables:
        logger.warning("No tables provided for relationship analysis")
        return {'error': 'No tables provided'}
    
    # Apply configuration overrides if provided
    if config:
        Config.update_from_dict(config)
    
    logger.info(f"Analyzing relationships between {len(tables)} tables")
    
    # First, detect primary keys for all tables
    primary_keys = {}
    for table_name, df in tables.items():
        try:
            pk_candidates = detect_primary_keys_detailed(df, table_name)
            best_pk = get_best_primary_key_candidate(pk_candidates)
            
            primary_keys[table_name] = {
                'candidates': pk_candidates,
                'best_candidate': best_pk,
                'has_strong_pk': best_pk is not None and best_pk['score'] > 0.8
            }
            
            if best_pk:
                logger.info(f"Best PK for {table_name}: {best_pk['column']} (score: {best_pk['score']:.3f})")
            else:
                logger.warning(f"No strong PK candidate found for {table_name}")
        
        except Exception as e:
            logger.error(f"Error detecting PK for table {table_name}: {e}")
            primary_keys[table_name] = {'error': str(e)}
    
    # Next, detect foreign key relationships between all table pairs
    foreign_key_relationships = []
    
    table_names = list(tables.keys())
    for i, fk_table in enumerate(table_names):
        for j, pk_table in enumerate(table_names):
            if i == j:  # Skip self-relationships for now
                continue
            
            try:
                fk_df = tables[fk_table]
                pk_df = tables[pk_table]
                
                # Get PK candidates for the target table
                pk_info = primary_keys.get(pk_table, {})
                pk_candidates = []
                
                if 'candidates' in pk_info:
                    # Use all reasonably scored PK candidates
                    pk_candidates = [
                        c['column'] for c in pk_info['candidates'] 
                        if c['score'] > 0.5
                    ]
                
                if not pk_candidates:
                    # Fall back to all columns if no good PK candidates
                    pk_candidates = list(pk_df.columns)
                
                # Detect FK relationships
                fk_relationships = detect_foreign_keys_detailed(
                    fk_df, fk_table, pk_df, pk_table, pk_candidates
                )
                
                # Filter to best relationships
                best_relationships = filter_best_fk_relationships(fk_relationships, max_per_fk_column=2)
                
                # Add to global list
                foreign_key_relationships.extend(best_relationships)
                
                if best_relationships:
                    logger.info(f"Found {len(best_relationships)} FK relationships: "
                              f"{fk_table} -> {pk_table}")
            
            except Exception as e:
                logger.error(f"Error detecting FK relationships {fk_table} -> {pk_table}: {e}")
                continue
    
    # Filter to accepted relationships only for summary
    accepted_fk_relationships = [r for r in foreign_key_relationships if r['decision'] == 'accepted']
    
    # Analyze relationship network
    network_analysis = analyze_relationship_network(accepted_fk_relationships, list(tables.keys()))
    
    # Generate table profiles if requested
    table_profiles = {}
    for table_name, df in tables.items():
        try:
            profile = profile_detailed(df, table_name, include_relationships=False)
            table_profiles[table_name] = profile
        except Exception as e:
            logger.error(f"Error profiling table {table_name}: {e}")
            table_profiles[table_name] = {'error': str(e)}
    
    # Compile comprehensive analysis
    analysis_result = {
        'analysis_metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'table_count': len(tables),
            'total_relationships_found': len(foreign_key_relationships),
            'accepted_relationships': len(accepted_fk_relationships),
            'config_version': Config.version
        },
        'tables': {
            table_name: {
                'row_count': len(df),
                'column_count': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            for table_name, df in tables.items()
        },
        'primary_keys': primary_keys,
        'foreign_key_relationships': foreign_key_relationships,
        'accepted_relationships_only': accepted_fk_relationships,
        'network_analysis': network_analysis,
        'table_profiles': table_profiles,
        'relationship_summary': generate_relationship_summary(
            primary_keys, accepted_fk_relationships, list(tables.keys())
        )
    }
    
    logger.info(f"Relationship analysis complete: {len(accepted_fk_relationships)} "
                f"accepted relationships found across {len(tables)} tables")
    
    return analysis_result


def analyze_relationship_network(relationships: List[Dict[str, Any]], 
                               table_names: List[str]) -> Dict[str, Any]:
    """
    Analyze the network structure of table relationships.
    
    Args:
        relationships: List of FK relationships
        table_names: List of all table names
        
    Returns:
        Network analysis results
    """
    if not relationships:
        return {
            'node_count': len(table_names),
            'edge_count': 0,
            'connected_components': len(table_names),  # All isolated
            'graph_density': 0.0,
            'hub_tables': [],
            'leaf_tables': table_names.copy(),
            'isolated_tables': table_names.copy()
        }
    
    # Build adjacency information
    in_degree = {table: 0 for table in table_names}  # FK references pointing to this table
    out_degree = {table: 0 for table in table_names}  # FK references from this table
    
    edges = []
    connected_tables = set()
    
    for rel in relationships:
        fk_table = rel['fk_table']
        pk_table = rel['pk_table']
        
        out_degree[fk_table] += 1
        in_degree[pk_table] += 1
        
        edges.append((fk_table, pk_table))
        connected_tables.add(fk_table)
        connected_tables.add(pk_table)
    
    # Identify table roles
    hub_tables = []  # High in-degree (referenced by many)
    leaf_tables = []  # Zero out-degree (don't reference others)
    root_tables = []  # Zero in-degree (not referenced by others)
    isolated_tables = [t for t in table_names if t not in connected_tables]
    
    for table in table_names:
        if in_degree[table] >= 2:  # Referenced by 2+ tables
            hub_tables.append({
                'table': table,
                'in_degree': in_degree[table],
                'out_degree': out_degree[table]
            })
        
        if out_degree[table] == 0 and table in connected_tables:
            leaf_tables.append(table)
        
        if in_degree[table] == 0 and table in connected_tables:
            root_tables.append(table)
    
    # Calculate graph metrics
    total_possible_edges = len(table_names) * (len(table_names) - 1)
    graph_density = len(edges) / total_possible_edges if total_possible_edges > 0 else 0.0
    
    # Estimate connected components (simplified)
    connected_component_count = len(isolated_tables) + (1 if connected_tables else 0)
    
    return {
        'node_count': len(table_names),
        'edge_count': len(edges),
        'connected_tables': len(connected_tables),
        'isolated_tables': isolated_tables,
        'connected_components': connected_component_count,
        'graph_density': graph_density,
        'hub_tables': hub_tables,
        'leaf_tables': leaf_tables,
        'root_tables': root_tables,
        'degree_distribution': {
            'in_degree': dict(in_degree),
            'out_degree': dict(out_degree)
        },
        'edges': edges
    }


def generate_relationship_summary(primary_keys: Dict[str, Any],
                                relationships: List[Dict[str, Any]],
                                table_names: List[str]) -> Dict[str, Any]:
    """
    Generate a high-level summary of the relationship analysis.
    
    Args:
        primary_keys: Primary key analysis results
        relationships: FK relationships
        table_names: List of table names
        
    Returns:
        Relationship summary
    """
    summary = {
        'total_tables': len(table_names),
        'tables_with_strong_pk': 0,
        'tables_without_pk': 0,
        'total_relationships': len(relationships),
        'relationship_strength_distribution': {
            'strong': 0,    # score > 0.8
            'medium': 0,    # 0.5 < score <= 0.8
            'weak': 0       # score <= 0.5
        },
        'coverage_analysis': {
            'tables_as_pk': set(),
            'tables_as_fk': set(),
            'orphaned_tables': set(table_names)
        }
    }
    
    # Analyze primary keys
    for table_name, pk_info in primary_keys.items():
        if pk_info.get('has_strong_pk', False):
            summary['tables_with_strong_pk'] += 1
        elif not pk_info.get('best_candidate'):
            summary['tables_without_pk'] += 1
    
    # Analyze relationships
    for rel in relationships:
        score = rel['score']
        if score > 0.8:
            summary['relationship_strength_distribution']['strong'] += 1
        elif score > 0.5:
            summary['relationship_strength_distribution']['medium'] += 1
        else:
            summary['relationship_strength_distribution']['weak'] += 1
        
        # Track table coverage
        summary['coverage_analysis']['tables_as_pk'].add(rel['pk_table'])
        summary['coverage_analysis']['tables_as_fk'].add(rel['fk_table'])
        
        # Remove from orphaned if involved in relationships
        summary['coverage_analysis']['orphaned_tables'].discard(rel['pk_table'])
        summary['coverage_analysis']['orphaned_tables'].discard(rel['fk_table'])
    
    # Convert sets to lists for JSON serialization
    summary['coverage_analysis']['tables_as_pk'] = list(summary['coverage_analysis']['tables_as_pk'])
    summary['coverage_analysis']['tables_as_fk'] = list(summary['coverage_analysis']['tables_as_fk'])
    summary['coverage_analysis']['orphaned_tables'] = list(summary['coverage_analysis']['orphaned_tables'])
    
    # Add insights
    insights = []
    
    if summary['tables_without_pk'] > 0:
        insights.append(f"{summary['tables_without_pk']} tables lack strong primary keys")
    
    if summary['relationship_strength_distribution']['strong'] > 0:
        insights.append(f"{summary['relationship_strength_distribution']['strong']} strong relationships found")
    
    if len(summary['coverage_analysis']['orphaned_tables']) > 0:
        insights.append(f"{len(summary['coverage_analysis']['orphaned_tables'])} tables are not connected to others")
    
    total_strong_medium = (summary['relationship_strength_distribution']['strong'] + 
                          summary['relationship_strength_distribution']['medium'])
    if total_strong_medium == summary['total_relationships'] and summary['total_relationships'] > 0:
        insights.append("All detected relationships are of good quality")
    
    summary['insights'] = insights
    
    return summary


def map_relationships_detailed(tables: Dict[str, pd.DataFrame], 
                             output_format: str = 'comprehensive',
                             config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main function to map relationships between tables with detailed analysis.
    
    Args:
        tables: Dictionary mapping table names to DataFrames
        output_format: Output format ('comprehensive', 'summary', 'relationships_only')
        config_overrides: Optional configuration overrides
        
    Returns:
        Relationship mapping results
    """
    if not tables:
        return {'error': 'No tables provided', 'tables': {}}
    
    # Validate inputs
    valid_tables = {}
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            valid_tables[name] = df
        else:
            logger.warning(f"Skipping invalid/empty table: {name}")
    
    if not valid_tables:
        return {'error': 'No valid tables provided', 'tables': {}}
    
    # Perform comprehensive analysis
    analysis = analyze_table_relationships(valid_tables, config_overrides)
    
    # Format output based on requested format
    if output_format == 'summary':
        return {
            'format': 'summary',
            'analysis_metadata': analysis['analysis_metadata'],
            'relationship_summary': analysis['relationship_summary'],
            'network_analysis': analysis['network_analysis'],
            'accepted_relationships': [
                {
                    'fk_table': r['fk_table'],
                    'fk_column': r['fk_column'],
                    'pk_table': r['pk_table'],
                    'pk_column': r['pk_column'],
                    'score': r['score'],
                    'relationship_type': r['metadata']['relationship_type']
                }
                for r in analysis['accepted_relationships_only']
            ]
        }
    
    elif output_format == 'relationships_only':
        return {
            'format': 'relationships_only',
            'foreign_key_relationships': analysis['foreign_key_relationships'],
            'accepted_relationships': analysis['accepted_relationships_only'],
            'total_found': analysis['analysis_metadata']['total_relationships_found'],
            'total_accepted': analysis['analysis_metadata']['accepted_relationships']
        }
    
    else:  # comprehensive (default)
        return analysis


def visualize_relationships(analysis: Dict[str, Any], 
                          output_path: Optional[str] = None) -> Optional[str]:
    """
    Generate a textual visualization of the relationship network.
    
    Args:
        analysis: Relationship analysis results
        output_path: Optional file path to save visualization
        
    Returns:
        Visualization string or file path if saved
    """
    if 'accepted_relationships_only' not in analysis:
        return "No relationship data available for visualization"
    
    relationships = analysis['accepted_relationships_only']
    network = analysis.get('network_analysis', {})
    
    viz_lines = [
        "# Database Relationship Map",
        "",
        f"**Analysis Date:** {analysis.get('analysis_metadata', {}).get('timestamp', 'Unknown')}",
        f"**Tables:** {analysis.get('analysis_metadata', {}).get('table_count', 0)}",
        f"**Relationships:** {len(relationships)}",
        "",
        "## Table Network Structure",
        ""
    ]
    
    # Network overview
    if network:
        viz_lines.extend([
            f"- **Connected Tables:** {network.get('connected_tables', 0)}",
            f"- **Isolated Tables:** {len(network.get('isolated_tables', []))}",
            f"- **Graph Density:** {network.get('graph_density', 0):.1%}",
            ""
        ])
        
        # Hub tables
        hub_tables = network.get('hub_tables', [])
        if hub_tables:
            viz_lines.extend([
                "### Hub Tables (Referenced by Multiple Tables)",
                ""
            ])
            for hub in hub_tables:
                viz_lines.append(f"- **{hub['table']}**: {hub['in_degree']} incoming references")
            viz_lines.append("")
        
        # Isolated tables
        isolated = network.get('isolated_tables', [])
        if isolated:
            viz_lines.extend([
                "### Isolated Tables (No Relationships)",
                ""
            ])
            for table in isolated:
                viz_lines.append(f"- {table}")
            viz_lines.append("")
    
    # Relationship details
    if relationships:
        viz_lines.extend([
            "## Relationship Details",
            "",
            "| FK Table | FK Column | PK Table | PK Column | Score | Type |",
            "|----------|-----------|----------|-----------|-------|------|"
        ])
        
        for rel in sorted(relationships, key=lambda x: x['score'], reverse=True):
            rel_type = rel.get('metadata', {}).get('relationship_type', 'unknown')
            viz_lines.append(
                f"| {rel['fk_table']} | {rel['fk_column']} | "
                f"{rel['pk_table']} | {rel['pk_column']} | "
                f"{rel['score']:.3f} | {rel_type} |"
            )
        
        viz_lines.extend(["", "## Relationship Explanations", ""])
        
        for i, rel in enumerate(relationships[:10], 1):  # Limit to top 10
            explanation = rel.get('explanation', 'No explanation available')
            viz_lines.extend([
                f"### {i}. {rel['fk_table']}.{rel['fk_column']} → {rel['pk_table']}.{rel['pk_column']}",
                f"**Score:** {rel['score']:.3f}",
                f"**Explanation:** {explanation}",
                ""
            ])
    
    visualization = "\n".join(viz_lines)
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(visualization)
            logger.info(f"Relationship visualization saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving visualization to {output_path}: {e}")
            return visualization
    
    return visualization


def export_relationships_csv(relationships: List[Dict[str, Any]], 
                           output_path: str) -> bool:
    """
    Export relationships to CSV format.
    
    Args:
        relationships: List of relationship dictionaries
        output_path: Path to save CSV file
        
    Returns:
        Success status
    """
    try:
        # Prepare data for CSV
        csv_data = []
        for rel in relationships:
            csv_data.append({
                'fk_table': rel['fk_table'],
                'fk_column': rel['fk_column'],
                'pk_table': rel['pk_table'],
                'pk_column': rel['pk_column'],
                'score': rel['score'],
                'decision': rel['decision'],
                'relationship_type': rel.get('metadata', {}).get('relationship_type', 'unknown'),
                'fk_to_pk_coverage': rel.get('features', {}).get('fk_to_pk_coverage', 0),
                'constraint_violations': rel.get('constraint_analysis', {}).get('violation_count', 0),
                'name_similarity': rel.get('features', {}).get('name_similarity', 0),
                'explanation': rel.get('explanation', '')
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Relationships exported to CSV: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting relationships to CSV: {e}")
        return False