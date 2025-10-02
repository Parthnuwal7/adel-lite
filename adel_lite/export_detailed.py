"""
Standardized export schema for Adel-Lite enhanced analysis.
Provides unified JSON format with comprehensive metadata.
"""

import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from .config import Config
from .profile_detailed import profile_detailed
from .map_relationships_detailed import analyze_table_relationships

logger = logging.getLogger(__name__)


class SchemaExporter:
    """
    Export comprehensive schema analysis in standardized format.
    """
    
    def __init__(self, version: str = "0.2.0-detailed"):
        """Initialize exporter with version info."""
        self.version = version
        self.metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "adel_lite_version": version,
            "export_format_version": "1.0"
        }
    
    def export_schema_graph_detailed(self, 
                                   tables: Dict[str, pd.DataFrame],
                                   config_overrides: Optional[Dict[str, Any]] = None,
                                   include_samples: bool = False,
                                   sampling_strategy: str = "adaptive") -> Dict[str, Any]:
        """
        Export unified enriched schema graph.
        
        Args:
            tables: Dictionary mapping table names to DataFrames
            config_overrides: Optional configuration overrides
            include_samples: Whether to include sample data
            sampling_strategy: Sampling strategy to use
            
        Returns:
            Comprehensive schema analysis in standardized format
        """
        logger.info(f"Exporting schema graph for {len(tables)} tables")
        
        # Update metadata
        export_metadata = self.metadata.copy()
        export_metadata.update({
            "table_count": len(tables),
            "sampling_strategy": sampling_strategy,
            "config_overrides": config_overrides or {},
            "include_samples": include_samples
        })
        
        # Perform comprehensive relationship analysis
        relationship_analysis = analyze_table_relationships(tables, config_overrides)
        
        # Generate detailed table profiles
        table_profiles = self._generate_table_profiles(tables, include_samples, sampling_strategy)
        
        # Extract relationship information
        relationships = self._extract_relationships(relationship_analysis)
        
        # Build unified schema
        schema_graph = {
            "metadata": export_metadata,
            "configuration": self._extract_configuration(),
            "tables": table_profiles,
            "relationships": relationships,
            "network_analysis": relationship_analysis.get('network_analysis', {}),
            "summary": self._generate_summary(table_profiles, relationships),
            "quality_metrics": self._calculate_quality_metrics(table_profiles, relationships)
        }
        
        logger.info(f"Schema graph exported: {len(table_profiles)} tables, {len(relationships)} relationships")
        return schema_graph
    
    def _generate_table_profiles(self, tables: Dict[str, pd.DataFrame], 
                               include_samples: bool = False,
                               sampling_strategy: str = "adaptive") -> Dict[str, Any]:
        """Generate detailed profiles for all tables."""
        profiles = {}
        
        for table_name, df in tables.items():
            try:
                # Generate comprehensive profile
                profile = profile_detailed(df, table_name, include_relationships=False)
                
                # Add table-level metadata
                table_metadata = {
                    "table_name": table_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                    "sampling_applied": len(df) > Config.sample_size,
                    "sampling_strategy": sampling_strategy if len(df) > Config.sample_size else None
                }
                
                # Add sample data if requested
                if include_samples:
                    sample_size = min(5, len(df))
                    table_metadata["sample_data"] = df.head(sample_size).to_dict('records')
                
                # Combine profile with metadata
                enhanced_profile = {
                    "metadata": table_metadata,
                    "profile": profile
                }
                
                profiles[table_name] = enhanced_profile
                
            except Exception as e:
                logger.error(f"Error profiling table {table_name}: {e}")
                profiles[table_name] = {
                    "metadata": {"table_name": table_name, "error": str(e)},
                    "profile": {}
                }
        
        return profiles
    
    def _extract_relationships(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and format relationship information."""
        relationships = []
        
        # Get all relationships (not just accepted ones)
        all_relationships = analysis.get('foreign_key_relationships', [])
        
        for rel in all_relationships:
            try:
                relationship = {
                    "relationship_id": f"{rel['fk_table']}.{rel['fk_column']}->{rel['pk_table']}.{rel['pk_column']}",
                    "foreign_table": rel['fk_table'],
                    "foreign_column": rel['fk_column'],
                    "referenced_table": rel['pk_table'],
                    "referenced_column": rel['pk_column'],
                    "score": rel['score'],
                    "decision": rel['decision'],
                    "confidence_level": self._get_confidence_level(rel['score']),
                    "explanation": rel.get('explanation', ''),
                    "detailed_explanation": rel.get('detailed_explanation', ''),
                    "features": rel.get('features', {}),
                    "metadata": {
                        "relationship_type": rel.get('metadata', {}).get('relationship_type', 'unknown'),
                        "constraint_analysis": rel.get('constraint_analysis', {}),
                        "coverage_analysis": rel.get('coverage_analysis', {})
                    }
                }
                
                relationships.append(relationship)
                
            except Exception as e:
                logger.error(f"Error formatting relationship: {e}")
                continue
        
        # Sort by score (highest first)
        relationships.sort(key=lambda x: x['score'], reverse=True)
        
        return relationships
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return "very_high"
        elif score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _extract_configuration(self) -> Dict[str, Any]:
        """Extract current configuration settings."""
        return {
            "pk_uniqueness_threshold": Config.pk_uniqueness_threshold,
            "fk_coverage_threshold": Config.fk_coverage_threshold,
            "name_similarity_threshold": Config.name_similarity_threshold,
            "decision_accept_threshold": Config.decision_accept_threshold,
            "decision_reject_threshold": Config.decision_reject_threshold,
            "sample_size": Config.sample_size,
            "scoring_weights": Config.scoring_weights,
            "pattern_configs": Config.pattern_configs,
            "version": Config.version
        }
    
    def _generate_summary(self, table_profiles: Dict[str, Any], 
                         relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level summary statistics."""
        # Count tables by characteristics
        tables_with_strong_pk = 0
        tables_without_pk = 0
        total_columns = 0
        total_rows = 0
        
        for table_name, profile_data in table_profiles.items():
            profile = profile_data.get('profile', {})
            metadata = profile_data.get('metadata', {})
            
            total_rows += metadata.get('row_count', 0)
            total_columns += metadata.get('column_count', 0)
            
            # Check for strong PK candidates
            primary_keys = profile.get('primary_key_candidates', [])
            if any(pk.get('score', 0) > 0.8 for pk in primary_keys):
                tables_with_strong_pk += 1
            elif not primary_keys:
                tables_without_pk += 1
        
        # Count relationships by decision
        relationship_counts = {
            "accepted": len([r for r in relationships if r['decision'] == 'accepted']),
            "rejected": len([r for r in relationships if r['decision'] == 'rejected']),
            "ambiguous": len([r for r in relationships if r['decision'] == 'ambiguous'])
        }
        
        # Count relationships by confidence
        confidence_counts = {
            "very_high": len([r for r in relationships if r['confidence_level'] == 'very_high']),
            "high": len([r for r in relationships if r['confidence_level'] == 'high']),
            "medium": len([r for r in relationships if r['confidence_level'] == 'medium']),
            "low": len([r for r in relationships if r['confidence_level'] == 'low']),
            "very_low": len([r for r in relationships if r['confidence_level'] == 'very_low'])
        }
        
        return {
            "table_statistics": {
                "total_tables": len(table_profiles),
                "total_columns": total_columns,
                "total_rows": total_rows,
                "tables_with_strong_pk": tables_with_strong_pk,
                "tables_without_pk": tables_without_pk,
                "average_columns_per_table": total_columns / len(table_profiles) if table_profiles else 0
            },
            "relationship_statistics": {
                "total_relationships": len(relationships),
                "by_decision": relationship_counts,
                "by_confidence": confidence_counts,
                "acceptance_rate": relationship_counts["accepted"] / len(relationships) if relationships else 0
            }
        }
    
    def _calculate_quality_metrics(self, table_profiles: Dict[str, Any], 
                                 relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        quality_metrics = {
            "schema_completeness": 0.0,
            "relationship_quality": 0.0,
            "primary_key_coverage": 0.0,
            "data_consistency": 0.0,
            "overall_quality_score": 0.0
        }
        
        if not table_profiles:
            return quality_metrics
        
        # Calculate primary key coverage
        tables_with_pk = 0
        for table_name, profile_data in table_profiles.items():
            profile = profile_data.get('profile', {})
            primary_keys = profile.get('primary_key_candidates', [])
            if any(pk.get('score', 0) > 0.5 for pk in primary_keys):
                tables_with_pk += 1
        
        quality_metrics["primary_key_coverage"] = tables_with_pk / len(table_profiles)
        
        # Calculate relationship quality
        if relationships:
            avg_relationship_score = sum(r['score'] for r in relationships) / len(relationships)
            quality_metrics["relationship_quality"] = avg_relationship_score
        
        # Calculate schema completeness (tables with both PK and relationships)
        connected_tables = set()
        for rel in relationships:
            if rel['decision'] == 'accepted':
                connected_tables.add(rel['foreign_table'])
                connected_tables.add(rel['referenced_table'])
        
        quality_metrics["schema_completeness"] = len(connected_tables) / len(table_profiles)
        
        # Calculate data consistency (based on constraint violations)
        total_violations = 0
        total_checks = 0
        for rel in relationships:
            constraint_analysis = rel.get('metadata', {}).get('constraint_analysis', {})
            if 'violation_count' in constraint_analysis and 'total_fk_values' in constraint_analysis:
                total_violations += constraint_analysis['violation_count']
                total_checks += constraint_analysis['total_fk_values']
        
        quality_metrics["data_consistency"] = 1.0 - (total_violations / total_checks) if total_checks > 0 else 1.0
        
        # Calculate overall quality score
        scores = [
            quality_metrics["schema_completeness"],
            quality_metrics["relationship_quality"],
            quality_metrics["primary_key_coverage"],
            quality_metrics["data_consistency"]
        ]
        quality_metrics["overall_quality_score"] = sum(scores) / len(scores)
        
        return quality_metrics
    
    def save_to_file(self, schema_graph: Dict[str, Any], 
                    filepath: str, format: str = "json") -> bool:
        """
        Save schema graph to file.
        
        Args:
            schema_graph: Schema graph data
            filepath: Output file path
            format: Output format ('json', 'yaml')
            
        Returns:
            Success status
        """
        try:
            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(schema_graph, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == "yaml":
                import yaml
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(schema_graph, f, default_flow_style=False, allow_unicode=True)
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Schema graph saved to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving schema graph: {e}")
            return False
    
    def export_summary_report(self, schema_graph: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            schema_graph: Schema graph data
            
        Returns:
            Markdown-formatted summary report
        """
        metadata = schema_graph.get('metadata', {})
        summary = schema_graph.get('summary', {})
        quality = schema_graph.get('quality_metrics', {})
        relationships = schema_graph.get('relationships', [])
        
        report_lines = [
            "# Database Schema Analysis Report",
            "",
            f"**Generated:** {metadata.get('created_at', 'Unknown')}",
            f"**Adel-Lite Version:** {metadata.get('adel_lite_version', 'Unknown')}",
            f"**Tables Analyzed:** {metadata.get('table_count', 0)}",
            "",
            "## Summary Statistics",
            ""
        ]
        
        # Table statistics
        table_stats = summary.get('table_statistics', {})
        report_lines.extend([
            "### Tables",
            f"- **Total Tables:** {table_stats.get('total_tables', 0)}",
            f"- **Total Columns:** {table_stats.get('total_columns', 0)}",
            f"- **Total Rows:** {table_stats.get('total_rows', 0):,}",
            f"- **Tables with Strong PK:** {table_stats.get('tables_with_strong_pk', 0)}",
            f"- **Tables without PK:** {table_stats.get('tables_without_pk', 0)}",
            ""
        ])
        
        # Relationship statistics
        rel_stats = summary.get('relationship_statistics', {})
        report_lines.extend([
            "### Relationships",
            f"- **Total Relationships Found:** {rel_stats.get('total_relationships', 0)}",
            f"- **Accepted:** {rel_stats.get('by_decision', {}).get('accepted', 0)}",
            f"- **Rejected:** {rel_stats.get('by_decision', {}).get('rejected', 0)}",
            f"- **Ambiguous:** {rel_stats.get('by_decision', {}).get('ambiguous', 0)}",
            f"- **Acceptance Rate:** {rel_stats.get('acceptance_rate', 0):.1%}",
            ""
        ])
        
        # Quality metrics
        report_lines.extend([
            "## Quality Metrics",
            "",
            f"- **Overall Quality Score:** {quality.get('overall_quality_score', 0):.1%}",
            f"- **Primary Key Coverage:** {quality.get('primary_key_coverage', 0):.1%}",
            f"- **Schema Completeness:** {quality.get('schema_completeness', 0):.1%}",
            f"- **Relationship Quality:** {quality.get('relationship_quality', 0):.1%}",
            f"- **Data Consistency:** {quality.get('data_consistency', 0):.1%}",
            ""
        ])
        
        # Top relationships
        accepted_relationships = [r for r in relationships if r['decision'] == 'accepted']
        if accepted_relationships:
            report_lines.extend([
                "## Top Relationships",
                "",
                "| Foreign Table | Foreign Column | Referenced Table | Referenced Column | Score | Confidence |",
                "|---------------|----------------|------------------|-------------------|-------|------------|"
            ])
            
            for rel in accepted_relationships[:10]:  # Top 10
                report_lines.append(
                    f"| {rel['foreign_table']} | {rel['foreign_column']} | "
                    f"{rel['referenced_table']} | {rel['referenced_column']} | "
                    f"{rel['score']:.3f} | {rel['confidence_level']} |"
                )
        
        return "\n".join(report_lines)


def export_schema_graph_detailed(tables: Dict[str, pd.DataFrame],
                                config_overrides: Optional[Dict[str, Any]] = None,
                                include_samples: bool = False,
                                sampling_strategy: str = "adaptive",
                                output_path: Optional[str] = None,
                                output_format: str = "json") -> Dict[str, Any]:
    """
    Convenient function to export schema graph with optional file output.
    
    Args:
        tables: Dictionary mapping table names to DataFrames
        config_overrides: Optional configuration overrides
        include_samples: Whether to include sample data
        sampling_strategy: Sampling strategy to use
        output_path: Optional file path to save results
        output_format: Output format ('json', 'yaml')
        
    Returns:
        Comprehensive schema analysis
    """
    exporter = SchemaExporter()
    schema_graph = exporter.export_schema_graph_detailed(
        tables=tables,
        config_overrides=config_overrides,
        include_samples=include_samples,
        sampling_strategy=sampling_strategy
    )
    
    if output_path:
        success = exporter.save_to_file(schema_graph, output_path, output_format)
        if success:
            logger.info(f"Schema graph exported to {output_path}")
        else:
            logger.error(f"Failed to export schema graph to {output_path}")
    
    return schema_graph