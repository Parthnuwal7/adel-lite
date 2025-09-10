import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ColumnProfile:
    """Profile information for a single column"""
    name: str
    dtype: str
    unique_count: int
    total_count: int
    null_count: int
    uniqueness_ratio: float
    is_pk_candidate: bool
    sample_values: List[str]
    has_duplicates: bool

@dataclass
class CompositeKey:
    """Represents a composite key candidate"""
    columns: List[str]
    is_unique: bool
    null_combinations: int
    
@dataclass
class TableProfile:
    """Profile information for a table/dataframe"""
    name: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    primary_key_candidates: List[str]
    composite_key_candidates: List[CompositeKey]
    is_junction_table: bool
    junction_table_score: float

@dataclass
class Relationship:
    """Represents a detected relationship between tables"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    match_ratio: float
    match_count: int
    relationship_type: str
    is_foreign_key: bool
    confidence_score: float

class DataFrameProfiler:
    """  profiler with composite key detection"""
    
    def __init__(self, pk_threshold: float = 1.0, max_composite_size: int = 3, sample_size: int = 5):
        self.pk_threshold = pk_threshold
        self.max_composite_size = max_composite_size
        self.sample_size = sample_size
    
    def profile_column(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """Profile a single column with PK detection"""
        total_count = len(series)
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        has_duplicates = unique_count < (total_count - null_count)
        
        # Strict PK candidate criteria: perfect uniqueness + no nulls
        is_pk_candidate = (
            uniqueness_ratio == 1.0 and 
            null_count == 0 and 
            total_count > 0
        )
        
        # Get sample values (convert to string for JSON serialization)
        sample_values = series.dropna().head(self.sample_size).astype(str).tolist()
        
        return ColumnProfile(
            name=column_name,
            dtype=str(series.dtype),
            unique_count=unique_count,
            total_count=total_count,
            null_count=null_count,
            uniqueness_ratio=uniqueness_ratio,
            is_pk_candidate=is_pk_candidate,
            sample_values=sample_values,
            has_duplicates=has_duplicates
        )
    
    def detect_composite_keys(self, df: pd.DataFrame) -> List[CompositeKey]:
        """Detect composite key candidates"""
        composite_keys = []
        columns = df.columns.tolist()
        
        # Try combinations of 2 to max_composite_size columns
        for size in range(2, min(len(columns) + 1, self.max_composite_size + 1)):
            for col_combo in combinations(columns, size):
                # Check if combination is unique
                grouped = df.groupby(list(col_combo)).size()
                is_unique = grouped.max() == 1
                
                # Count null combinations
                null_mask = df[list(col_combo)].isnull().any(axis=1)
                null_combinations = null_mask.sum()
                
                # Only consider as composite key if unique and minimal nulls
                if is_unique and null_combinations == 0:
                    composite_keys.append(CompositeKey(
                        columns=list(col_combo),
                        is_unique=is_unique,
                        null_combinations=null_combinations
                    ))
        
        # Sort by smallest combination first (prefer simpler keys)
        composite_keys.sort(key=lambda x: len(x.columns))
        return composite_keys
    
    def calculate_junction_table_score(self, df: pd.DataFrame, 
                                     column_profiles: Dict[str, ColumnProfile]) -> float:
        """Calculate likelihood that this is a junction table"""
        if len(df.columns) == 0:
            return 0.0
            
        # Factors that suggest junction table:
        score = 0.0
        
        # 1. High percentage of columns could be FKs (low uniqueness but not PK)
        potential_fk_cols = [
            col for col, profile in column_profiles.items()
            if not profile.is_pk_candidate and profile.has_duplicates
        ]
        fk_ratio = len(potential_fk_cols) / len(df.columns)
        score += fk_ratio * 0.4
        
        # 2. Few non-FK attributes (prefer tables with mostly keys)
        non_key_cols = [
            col for col, profile in column_profiles.items()
            if not profile.is_pk_candidate and col not in potential_fk_cols
        ]
        if len(non_key_cols) <= 2:  # Very few non-key columns
            score += 0.3
            
        # 3. Multiple columns with high cardinality (could be FKs)
        high_cardinality_cols = [
            col for col, profile in column_profiles.items()
            if profile.unique_count > 10 and not profile.is_pk_candidate
        ]
        if len(high_cardinality_cols) >= 2:
            score += 0.2
            
        # 4. Small number of columns overall (junction tables are typically simple)
        if len(df.columns) <= 5:
            score += 0.1
            
        return min(score, 1.0)  # Cap at 1.0
    
    def profile_dataframe(self, df: pd.DataFrame, table_name: str) -> TableProfile:
        """Profile an entire dataframe with   detection"""
        columns = {}
        pk_candidates = []
        
        # Profile individual columns
        for col_name in df.columns:
            col_profile = self.profile_column(df[col_name], col_name)
            columns[col_name] = col_profile
            
            if col_profile.is_pk_candidate:
                pk_candidates.append(col_name)
        
        # Detect composite keys if no single-column PK found
        composite_keys = []
        if not pk_candidates:  # Only look for composite keys if no single PKs
            composite_keys = self.detect_composite_keys(df)
        
        # Calculate junction table likelihood
        junction_score = self.calculate_junction_table_score(df, columns)
        is_junction = junction_score > 0.6  # Threshold for junction table detection
        
        return TableProfile(
            name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            primary_key_candidates=pk_candidates,
            composite_key_candidates=composite_keys,
            is_junction_table=is_junction,
            junction_table_score=junction_score
        )

class RelationshipDetector:
    """  relationship detector focusing on FK->PK relationships"""
    
    def __init__(self, fk_threshold: float = 0.9, min_match_count: int = 5):
        self.fk_threshold = fk_threshold  # Higher threshold for FK detection
        self.min_match_count = min_match_count
    
    def check_fk_relationship(self, potential_fk_series: pd.Series, 
                            pk_series: pd.Series) -> Tuple[float, int, float]:
        """Check if a column is a foreign key to a primary key"""
        # Remove nulls from potential FK (nulls are allowed in FKs)
        fk_non_null = potential_fk_series.dropna()
        
        if len(fk_non_null) == 0:
            return 0.0, 0, 0.0
        
        # Convert to sets for faster intersection
        fk_values = set(fk_non_null.astype(str))
        pk_values = set(pk_series.dropna().astype(str))
        
        if not pk_values:
            return 0.0, 0, 0.0
        
        # Count matches
        matches = fk_values.intersection(pk_values)
        match_count = len(matches)
        
        # Calculate match ratio: matches / total non-null FK values
        match_ratio = match_count / len(fk_non_null) if len(fk_non_null) > 0 else 0.0
        
        # Calculate confidence based on various factors
        confidence = self.calculate_fk_confidence(
            match_ratio, match_count, len(fk_non_null), len(pk_values)
        )
        
        return match_ratio, match_count, confidence
    
    def calculate_fk_confidence(self, match_ratio: float, match_count: int,
                               fk_size: int, pk_size: int) -> float:
        """Calculate confidence score for FK relationship"""
        confidence = 0.0
        
        # High match ratio is most important
        confidence += match_ratio * 0.6
        
        # Sufficient absolute matches
        if match_count >= self.min_match_count:
            confidence += 0.2
        
        # FK size relative to PK size (good FKs often reference most PK values)
        if pk_size > 0:
            pk_coverage = match_count / pk_size
            confidence += pk_coverage * 0.2
        
        return min(confidence, 1.0)
    
    def determine_relationship_type(self, fk_column: pd.Series, pk_column: pd.Series,
                                   fk_profile: ColumnProfile) -> str:
        """Determine relationship type based on column characteristics"""
        # Check if FK column has duplicates
        fk_has_duplicates = fk_profile.has_duplicates
        
        if not fk_has_duplicates:
            # FK is also unique -> one-to-one
            return "one-to-one"
        else:
            # FK has duplicates, PK is unique -> many-to-one
            return "many-to-one"
    
    def detect_relationships(self, dataframes: Dict[str, pd.DataFrame], 
                           profiles: Dict[str, TableProfile]) -> List[Relationship]:
        """Detect FK relationships by comparing non-PK columns to PK candidates"""
        relationships = []
        
        # First, collect all PK candidates from all tables
        pk_candidates = {}  # {table_name: {column_name: series}}
        
        for table_name, profile in profiles.items():
            pk_candidates[table_name] = {}
            df = dataframes[table_name]
            
            # Add single-column PK candidates
            for pk_col in profile.primary_key_candidates:
                pk_candidates[table_name][pk_col] = df[pk_col]
        
        # Now check each non-PK column against all PK candidates
        for source_table, source_profile in profiles.items():
            source_df = dataframes[source_table]
            
            for col_name, col_profile in source_profile.columns.items():
                # Skip if this column is a PK candidate in its own table
                if col_name in source_profile.primary_key_candidates:
                    continue
                
                source_series = source_df[col_name]
                
                # Check against all PK candidates in other tables
                for target_table, target_pks in pk_candidates.items():
                    if target_table == source_table:  # Skip same table
                        continue
                    
                    for pk_col, pk_series in target_pks.items():
                        # Skip if data types are incompatible
                        if not self._are_compatible_types(source_series.dtype, pk_series.dtype):
                            continue
                        
                        match_ratio, match_count, confidence = self.check_fk_relationship(
                            source_series, pk_series
                        )
                        
                        # Determine if this is a valid FK relationship
                        is_fk = (match_ratio >= self.fk_threshold and 
                                match_count >= self.min_match_count and
                                confidence > 0.7)
                        
                        if match_ratio > 0.1:  # Store even weak relationships for analysis
                            rel_type = self.determine_relationship_type(
                                source_series, pk_series, col_profile
                            )
                            
                            relationships.append(Relationship(
                                source_table=source_table,
                                source_column=col_name,
                                target_table=target_table,
                                target_column=pk_col,
                                match_ratio=match_ratio,
                                match_count=match_count,
                                relationship_type=rel_type,
                                is_foreign_key=is_fk,
                                confidence_score=confidence
                            ))
        
        # Sort by confidence score (best relationships first)
        relationships.sort(key=lambda x: x.confidence_score, reverse=True)
        return relationships
    
    def _are_compatible_types(self, dtype1, dtype2) -> bool:
        """  type compatibility checking"""
        def categorize_dtype(dtype):
            dtype_str = str(dtype).lower()
            if any(t in dtype_str for t in ['int', 'int64', 'int32']):
                return 'integer'
            elif any(t in dtype_str for t in ['float', 'float64', 'float32']):
                return 'float'
            elif any(t in dtype_str for t in ['object', 'string']):
                return 'string'
            elif 'datetime' in dtype_str:
                return 'datetime'
            elif 'bool' in dtype_str:
                return 'boolean'
            else:
                return 'other'
        
        cat1, cat2 = categorize_dtype(dtype1), categorize_dtype(dtype2)
        
        # Strict compatibility for FK relationships
        compatible_pairs = [
            ('integer', 'integer'),
            ('string', 'string'),
            ('string', 'integer'),  # String can contain integer IDs
            ('integer', 'string'),  # Integer IDs can match string IDs
            ('datetime', 'datetime'),
            ('float', 'float'),
            ('boolean', 'boolean')
        ]
        
        return (cat1, cat2) in compatible_pairs

class SchemaGenerator:
    """  schema generator with junction table detection"""
    
    def generate_schema(self, profiles: Dict[str, TableProfile], 
                       relationships: List[Relationship]) -> Dict:
        """Generate   schema with junction table information"""
        schema = {
            "tables": {},
            "relationships": [],
            "junction_tables": [],
            "foreign_keys": [],
            "metadata": {
                "total_tables": len(profiles),
                "total_relationships": len(relationships),
                "foreign_key_relationships": len([r for r in relationships if r.is_foreign_key]),
                "junction_tables_detected": len([p for p in profiles.values() if p.is_junction_table]),
                "generation_timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        # Add table information
        for table_name, profile in profiles.items():
            table_info = {
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "primary_key_candidates": profile.primary_key_candidates,
                "composite_key_candidates": [
                    {"columns": ck.columns, "is_unique": ck.is_unique}
                    for ck in profile.composite_key_candidates
                ],
                "is_junction_table": profile.is_junction_table,
                "junction_table_score": round(profile.junction_table_score, 3),
                "columns": {
                    col_name: {
                        "dtype": col_profile.dtype,
                        "unique_count": col_profile.unique_count,
                        "uniqueness_ratio": round(col_profile.uniqueness_ratio, 3),
                        "null_count": col_profile.null_count,
                        "is_pk_candidate": col_profile.is_pk_candidate,
                        "has_duplicates": col_profile.has_duplicates,
                        "sample_values": col_profile.sample_values
                    }
                    for col_name, col_profile in profile.columns.items()
                }
            }
            
            schema["tables"][table_name] = table_info
            
            # Track junction tables separately
            if profile.is_junction_table:
                schema["junction_tables"].append({
                    "table_name": table_name,
                    "score": round(profile.junction_table_score, 3),
                    "potential_foreign_keys": [
                        col_name for col_name, col_profile in profile.columns.items()
                        if not col_profile.is_pk_candidate and col_profile.has_duplicates
                    ]
                })
        
        # Add relationship information
        for rel in relationships:
            rel_info = {
                "source_table": rel.source_table,
                "source_column": rel.source_column,
                "target_table": rel.target_table,
                "target_column": rel.target_column,
                "match_ratio": round(rel.match_ratio, 3),
                "match_count": rel.match_count,
                "relationship_type": rel.relationship_type,
                "is_foreign_key": rel.is_foreign_key,
                "confidence_score": round(rel.confidence_score, 3)
            }
            
            schema["relationships"].append(rel_info)
            
            # Track confirmed foreign keys separately
            if rel.is_foreign_key:
                schema["foreign_keys"].append(rel_info)
        
        return schema

class SchemaVisualizer:
    """  visualizer with junction table highlighting"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        self.figsize = figsize
    
    def create_graph(self, profiles: Dict[str, TableProfile], 
                    relationships: List[Relationship]) -> nx.Graph:
        """Create   graph with junction table information"""
        G = nx.Graph()
        
        # Add nodes with   attributes
        for table_name, profile in profiles.items():
            G.add_node(table_name, 
                      row_count=profile.row_count,
                      pk_candidates=len(profile.primary_key_candidates),
                      composite_keys=len(profile.composite_key_candidates),
                      is_junction=profile.is_junction_table,
                      junction_score=profile.junction_table_score)
        
        # Add edges for confirmed foreign keys only
        fk_relationships = [r for r in relationships if r.is_foreign_key]
        
        for rel in fk_relationships:
            edge_label = f"{rel.source_column} → {rel.target_column}\n({rel.match_ratio:.1%}, conf: {rel.confidence_score:.2f})"
            G.add_edge(rel.source_table, rel.target_table,
                      relationship=rel.relationship_type,
                      match_ratio=rel.match_ratio,
                      confidence=rel.confidence_score,
                      label=edge_label,
                      source_col=rel.source_column,
                      target_col=rel.target_column)
        
        return G
    
    def visualize_schema(self, profiles: Dict[str, TableProfile], 
                        relationships: List[Relationship], 
                        save_path: str = None) -> plt.Figure:
        """Create   ER diagram with junction table highlighting"""
        G = self.create_graph(profiles, relationships)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate layout with better spacing for junction tables
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=4, iterations=50, seed=42)
        else:
            pos = {}
        
        # Categorize nodes
        junction_tables = [node for node in G.nodes() if profiles[node].is_junction_table]
        pk_tables = [node for node in G.nodes() if profiles[node].primary_key_candidates and not profiles[node].is_junction_table]
        other_tables = [node for node in G.nodes() if not profiles[node].primary_key_candidates and not profiles[node].is_junction_table]
        
        # Draw junction tables (diamond shape)
        if junction_tables:
            junction_sizes = [profiles[node].row_count * 8 + 600 for node in junction_tables]
            nx.draw_networkx_nodes(G, pos, nodelist=junction_tables, 
                                  node_size=junction_sizes, 
                                  node_color='orange', node_shape='d', 
                                  alpha=0.8, ax=ax, label='Junction Tables')
        
        # Draw PK tables (circles)
        if pk_tables:
            pk_sizes = [profiles[node].row_count * 10 + 800 for node in pk_tables]
            nx.draw_networkx_nodes(G, pos, nodelist=pk_tables,
                                  node_size=pk_sizes,
                                  node_color='lightblue', node_shape='o',
                                  alpha=0.8, ax=ax, label='Tables with PKs')
        
        # Draw other tables (squares)
        if other_tables:
            other_sizes = [profiles[node].row_count * 10 + 600 for node in other_tables]
            nx.draw_networkx_nodes(G, pos, nodelist=other_tables,
                                  node_size=other_sizes,
                                  node_color='lightgray', node_shape='s',
                                  alpha=0.8, ax=ax, label='Other Tables')
        
        # Draw edges with different colors for confidence levels
        if G.edges():
            high_conf_edges = [(u, v) for u, v, d in G.edges(data=True) if d['confidence'] > 0.8]
            med_conf_edges = [(u, v) for u, v, d in G.edges(data=True) if 0.5 < d['confidence'] <= 0.8]
            low_conf_edges = [(u, v) for u, v, d in G.edges(data=True) if d['confidence'] <= 0.5]
            
            if high_conf_edges:
                nx.draw_networkx_edges(G, pos, edgelist=high_conf_edges,
                                      edge_color='darkgreen', width=3, alpha=0.8, ax=ax)
            if med_conf_edges:
                nx.draw_networkx_edges(G, pos, edgelist=med_conf_edges,
                                      edge_color='orange', width=2, alpha=0.7, ax=ax)
            if low_conf_edges:
                nx.draw_networkx_edges(G, pos, edgelist=low_conf_edges,
                                      edge_color='lightcoral', width=1, alpha=0.5, ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
        # Add simplified edge labels for high confidence relationships only
        high_conf_labels = {
            (u, v): f"{data['source_col']}→{data['target_col']}" 
            for u, v, data in G.edges(data=True) if data['confidence'] > 0.8
        }
        if high_conf_labels:
            nx.draw_networkx_edge_labels(G, pos, high_conf_labels, font_size=7, ax=ax)
        
        # Customize plot
        ax.set_title("  Auto-Detected Database Schema\n(Foreign Key Relationships Only)", 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        #   legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=12, label='Tables with Primary Keys'),
            plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='orange', 
                      markersize=12, label='Junction Tables'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
                      markersize=12, label='Other Tables'),
            plt.Line2D([0], [0], color='darkgreen', linewidth=3, label='High Confidence FK (>80%)'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='Medium Confidence FK (50-80%)'),
            plt.Line2D([0], [0], color='lightcoral', linewidth=1, label='Low Confidence FK (≤50%)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class AutoMapper:
    """  auto-mapper with improved PK/FK detection"""
    
    def __init__(self, pk_threshold: float = 1.0, fk_threshold: float = 0.9, 
                 max_composite_size: int = 3):
        self.profiler = DataFrameProfiler(
            pk_threshold=pk_threshold, 
            max_composite_size=max_composite_size
        )
        self.detector = RelationshipDetector(fk_threshold=fk_threshold)
        self.generator = SchemaGenerator()
        self.visualizer = SchemaVisualizer()
    
    def analyze_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """  analysis pipeline"""
        print("🔍   profiling of dataframes...")
        profiles = {}
        
        for table_name, df in dataframes.items():
            profiles[table_name] = self.profiler.profile_dataframe(df, table_name)
            profile = profiles[table_name]
            
            pk_info = f"{len(profile.primary_key_candidates)} single PKs"
            if profile.composite_key_candidates:
                pk_info += f", {len(profile.composite_key_candidates)} composite PKs"
            
            junction_info = f" (Junction: {profile.junction_table_score:.2f})" if profile.is_junction_table else ""
            
            print(f"   ✓ {table_name}: {len(df)} rows, {pk_info}{junction_info}")
        
        print("\n🔗   FK relationship detection...")
        relationships = self.detector.detect_relationships(dataframes, profiles)
        fk_relationships = [r for r in relationships if r.is_foreign_key]
        
        print(f"   ✓ Found {len(relationships)} potential relationships")
        print(f"   ✓ Confirmed {len(fk_relationships)} foreign key relationships")
        
        junction_tables = [name for name, profile in profiles.items() if profile.is_junction_table]
        if junction_tables:
            print(f"   ✓ Detected junction tables: {', '.join(junction_tables)}")
        
        print("\n📋 Generating   schema...")
        schema = self.generator.generate_schema(profiles, relationships)
        
        print("\n📊 Creating   visualization...")
        fig = self.visualizer.visualize_schema(profiles, relationships)
        
        return {
            "schema": schema,
            "profiles": profiles,
            "relationships": relationships,
            "foreign_key_relationships": fk_relationships,
            "visualization": fig
        }
    
    def print_detailed_summary(self, results: Dict):
        """Print detailed analysis summary"""
        schema = results["schema"]
        profiles = results["profiles"]
        
        print("\n" + "="*60)
        print("  ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Tables analyzed: {schema['metadata']['total_tables']}")
        print(f"🔗 Total relationships: {schema['metadata']['total_relationships']}")
        print(f"🔑 Confirmed foreign keys: {schema['metadata']['foreign_key_relationships']}")
        print(f"🔄 Junction tables: {schema['metadata']['junction_tables_detected']}")
        
        print("\n📋 Primary Key Analysis:")
        for table_name, table_info in schema["tables"].items():
            pk_candidates = table_info["primary_key_candidates"]
            composite_keys = table_info["composite_key_candidates"]
            
            if pk_candidates:
                print(f"   ✓ {table_name}: {pk_candidates}")
            elif composite_keys:
                comp_key_strs = [" + ".join(ck["columns"]) for ck in composite_keys[:2]]
                print(f"   ⚡ {table_name}: Composite keys: {comp_key_strs}")
            else:
                print(f"   ❌ {table_name}: No PK candidates found")
        
        print("\n🔑 Confirmed Foreign Key Relationships:")
        fk_rels = [r for r in results["relationships"] if r.is_foreign_key]
        for rel in fk_rels:
            print(f"   ✓ {rel.source_table}.{rel.source_column} → "
                  f"{rel.target_table}.{rel.target_column} "
                  f"({rel.match_ratio:.1%} match, {rel.confidence_score:.2f} confidence)")
        
        if schema["junction_tables"]:
            print("\n🔄 Junction Tables Detected:")
            for jt in schema["junction_tables"]:
                print(f"   🔄 {jt['table_name']} (score: {jt['score']:.2f})")
    
    def save_results(self, results: Dict, output_dir: str = "."):
        """Save   results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save   schema
        schema_path = os.path.join(output_dir, " _schema.json")
        with open(schema_path, 'w') as f:
            json.dump(results["schema"], f, indent=2, default=str)
        
        # Save FK relationships separately
        fk_path = os.path.join(output_dir, "foreign_keys.json")
        fk_data = {
            "foreign_key_relationships": [
                {
                    "source": f"{r.source_table}.{r.source_column}",
                    "target": f"{r.target_table}.{r.target_column}",
                    "match_ratio": r.match_ratio,
                    "confidence": r.confidence_score,
                    "relationship_type": r.relationship_type
                }
                for r in results["foreign_key_relationships"]
            ]
        }
        
        with open(fk_path, 'w') as f:
            json.dump(fk_data, f, indent=2)
        
        # Save visualization
        viz_path = os.path.join(output_dir, " _schema_diagram.png")
        results["visualization"].savefig(viz_path, dpi=300, bbox_inches='tight')
        
        print(f"\n💾   results saved:")
        print(f"   📄 Schema: {schema_path}")
        print(f"   🔑 Foreign Keys: {fk_path}")
        print(f"   🖼️  Diagram: {viz_path}")

# Example usage with   testing
if __name__ == "__main__":
    # Create   sample dataframes for testing
    
    # Users table (clean PK)
    users_df = pd.DataFrame({
        'user_id': range(1, 101),  # Perfect PK
        'username': [f'user_{i}' for i in range(1, 101)],  # Also unique
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'created_at': pd.date_range('2023-01-01', periods=100, freq='D')
    })
    
    # Products table (clean PK)
    products_df = pd.DataFrame({
        'product_id': range(1, 51),  # Perfect PK
        'product_name': [f'Product {i}' for i in range(1, 51)],
        'category_id': np.random.choice(range(1, 11), 50),  # FK to categories
        'price': np.random.uniform(5, 200, 50).round(2),
        'sku': [f'SKU-{i:04d}' for i in range(1, 51)]  # Another unique field
    })
    
    # Categories table (clean PK)
    categories_df = pd.DataFrame({
        'category_id': range(1, 11),  # Perfect PK
        'category_name': [f'Category {i}' for i in range(1, 11)],
        'description': [f'Description for category {i}' for i in range(1, 11)]
    })
    
    # Orders table (clean PK, multiple FKs)
    orders_df = pd.DataFrame({
        'order_id': range(1, 201),  # Perfect PK
        'user_id': np.random.choice(range(1, 101), 200),  # FK to users
        'order_date': pd.date_range('2023-01-01', periods=200, freq='12H'),
        'status': np.random.choice(['pending', 'shipped', 'delivered'], 200),
        'total_amount': np.random.uniform(20, 1000, 200).round(2)
    })
    
    # Order Items table (junction table - composite PK, multiple FKs)
    order_items_data = []
    for order_id in range(1, 201):
        # Each order has 1-4 items
        num_items = np.random.randint(1, 5)
        selected_products = np.random.choice(range(1, 51), size=num_items, replace=False)
        
        for product_id in selected_products:
            order_items_data.append({
                'order_id': order_id,  # FK to orders
                'product_id': int(product_id),  # FK to products
                'quantity': np.random.randint(1, 5),
                'unit_price': round(np.random.uniform(5, 200),2)
            })
    
    order_items_df = pd.DataFrame(order_items_data)
    
    # Reviews table (has FK to users and products)
    reviews_df = pd.DataFrame({
        'review_id': range(1, 151),  # Perfect PK
        'user_id': np.random.choice(range(1, 101), 150),  # FK to users
        'product_id': np.random.choice(range(1, 51), 150),  # FK to products
        'rating': np.random.randint(1, 6, 150),
        'review_text': [f'Review text {i}' for i in range(1, 151)],
        'created_at': pd.date_range('2023-02-01', periods=150, freq='6H')
    })
    
    # Customer Support table (composite PK example)
    support_tickets_df = pd.DataFrame({
        'ticket_date': pd.date_range('2023-01-01', periods=80, freq='2D'),
        'ticket_number': list(range(1, 21)) * 4,  # Combination makes it unique
        'user_id': np.random.choice(range(1, 101), 80),  # FK to users
        'issue_type': np.random.choice(['billing', 'technical', 'general'], 80),
        'status': np.random.choice(['open', 'closed', 'pending'], 80),
        'priority': np.random.choice(['low', 'medium', 'high'], 80)
    })
    
    # Run the   analysis
    dataframes = {
        'users': users_df,
        'products': products_df,
        'categories': categories_df,
        'orders': orders_df,
        'order_items': order_items_df,  # Should be detected as junction table
        'reviews': reviews_df,
        'support_tickets': support_tickets_df  # Should have composite PK
    }
    
    print("🚀 Starting   Auto-Mapping Analysis...")
    print("="*60)
    
    # Initialize   mapper with strict thresholds
    mapper = AutoMapper(
        pk_threshold=1.0,      # Perfect uniqueness required for PK
        fk_threshold=0.9,      # High match ratio required for FK
        max_composite_size=3   # Try up to 3-column composite keys
    )
    
    # Run analysis
    results = mapper.analyze_dataframes(dataframes)
    
    # Print detailed summary
    mapper.print_detailed_summary(results)

    print(SchemaGenerator().generate_schema(results["profiles"], results["relationships"]))
    
    # Show some weak relationships for analysis
    weak_relationships = [r for r in results["relationships"] 
                         if not r.is_foreign_key and r.match_ratio > 0.3]
    
    if weak_relationships:
        print(f"\n🔍 Potential relationships (not confirmed as FKs):")
        for rel in weak_relationships[:5]:  # Show top 5
            print(f"   ? {rel.source_table}.{rel.source_column} ~ "
                  f"{rel.target_table}.{rel.target_column} "
                  f"({rel.match_ratio:.1%} match, {rel.confidence_score:.2f} conf)")
    
    # Show the   visualization
    plt.show()
    
    # Optionally save results
    # mapper.save_results(results, " _output")