#!/usr/bin/env python3
"""
Enhanced Command Line Interface for Adel-Lite.
Supports detailed detection modes, configuration loading, and comprehensive output options.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from .map_relationships_detailed import map_relationships_detailed
from .export_detailed import export_schema_graph_detailed, SchemaExporter
from .config import Config, AdelLiteConfig
from .performance import PerformanceOptimizer, benchmark_performance
from .test_enhanced import run_all_tests


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_tables_from_csv(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load tables from CSV files."""
    tables = {}
    
    for file_path in file_paths:
        try:
            file_name = Path(file_path).stem
            df = pd.read_csv(file_path)
            tables[file_name] = df
            print(f"Loaded table '{file_name}' with {len(df)} rows, {len(df.columns)} columns")
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return tables


def load_configuration(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load configuration from file."""
    if not config_path:
        return None
    
    try:
        config = AdelLiteConfig.load_from_file(config_path)
        return config.to_dict()
    
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return None


def analyze_relationships(args):
    """Analyze relationships between tables."""
    print("=== Adel-Lite Enhanced Relationship Analysis ===\n")
    
    # Load configuration
    config_overrides = load_configuration(args.config) if args.config else None
    if config_overrides:
        print(f"Loaded configuration from {args.config}")
    
    # Load tables
    if args.csv_files:
        tables = load_tables_from_csv(args.csv_files)
    else:
        print("Error: No input files specified")
        return 1
    
    if not tables:
        print("Error: No tables loaded successfully")
        return 1
    
    # Perform analysis
    print(f"\nAnalyzing {len(tables)} tables...")
    
    if args.mode == 'detailed':
        # Use comprehensive detailed analysis
        schema_graph = export_schema_graph_detailed(
            tables=tables,
            config_overrides=config_overrides,
            include_samples=args.include_samples,
            sampling_strategy=args.sampling_strategy
        )
        
        # Save detailed results
        if args.output:
            exporter = SchemaExporter()
            success = exporter.save_to_file(schema_graph, args.output, args.format)
            if success:
                print(f"\nDetailed analysis saved to {args.output}")
            
            # Also save summary report
            if args.summary_report:
                summary_path = args.output.replace('.json', '_summary.md')
                summary = exporter.export_summary_report(schema_graph)
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary report saved to {summary_path}")
        
        # Print summary
        print_detailed_summary(schema_graph)
    
    else:
        # Use basic analysis
        results = map_relationships_detailed(
            tables=tables,
            output_format=args.output_format,
            config_overrides=config_overrides
        )
        
        # Save results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
        # Print basic summary
        print_basic_summary(results)
    
    return 0


def print_detailed_summary(schema_graph: Dict[str, Any]):
    """Print detailed analysis summary."""
    metadata = schema_graph.get('metadata', {})
    summary = schema_graph.get('summary', {})
    quality = schema_graph.get('quality_metrics', {})
    relationships = schema_graph.get('relationships', [])
    
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Table statistics
    table_stats = summary.get('table_statistics', {})
    print(f"\nTables:")
    print(f"  Total: {table_stats.get('total_tables', 0)}")
    print(f"  Total Rows: {table_stats.get('total_rows', 0):,}")
    print(f"  Total Columns: {table_stats.get('total_columns', 0)}")
    print(f"  With Strong PK: {table_stats.get('tables_with_strong_pk', 0)}")
    print(f"  Without PK: {table_stats.get('tables_without_pk', 0)}")
    
    # Relationship statistics
    rel_stats = summary.get('relationship_statistics', {})
    print(f"\nRelationships:")
    print(f"  Total Found: {rel_stats.get('total_relationships', 0)}")
    print(f"  Accepted: {rel_stats.get('by_decision', {}).get('accepted', 0)}")
    print(f"  Rejected: {rel_stats.get('by_decision', {}).get('rejected', 0)}")
    print(f"  Ambiguous: {rel_stats.get('by_decision', {}).get('ambiguous', 0)}")
    print(f"  Acceptance Rate: {rel_stats.get('acceptance_rate', 0):.1%}")
    
    # Quality metrics
    print(f"\nQuality Metrics:")
    print(f"  Overall Score: {quality.get('overall_quality_score', 0):.1%}")
    print(f"  PK Coverage: {quality.get('primary_key_coverage', 0):.1%}")
    print(f"  Schema Completeness: {quality.get('schema_completeness', 0):.1%}")
    print(f"  Relationship Quality: {quality.get('relationship_quality', 0):.1%}")
    print(f"  Data Consistency: {quality.get('data_consistency', 0):.1%}")
    
    # Top relationships
    accepted_rels = [r for r in relationships if r['decision'] == 'accepted']
    if accepted_rels:
        print(f"\nTop Relationships:")
        for i, rel in enumerate(accepted_rels[:5], 1):
            print(f"  {i}. {rel['foreign_table']}.{rel['foreign_column']} → "
                  f"{rel['referenced_table']}.{rel['referenced_column']} "
                  f"(score: {rel['score']:.3f})")


def print_basic_summary(results: Dict[str, Any]):
    """Print basic analysis summary."""
    print(f"\n{'='*50}")
    print("ANALYSIS RESULTS")
    print(f"{'='*50}")
    
    metadata = results.get('analysis_metadata', {})
    print(f"\nTables analyzed: {metadata.get('table_count', 0)}")
    print(f"Relationships found: {metadata.get('total_relationships_found', 0)}")
    print(f"Accepted relationships: {metadata.get('accepted_relationships', 0)}")
    
    # Show accepted relationships
    accepted_rels = results.get('accepted_relationships_only', [])
    if accepted_rels:
        print(f"\nAccepted Relationships:")
        for rel in accepted_rels:
            print(f"  {rel['fk_table']}.{rel['fk_column']} → "
                  f"{rel['pk_table']}.{rel['pk_column']} "
                  f"(score: {rel['score']:.3f})")


def benchmark_command(args):
    """Run performance benchmarks."""
    print("=== Adel-Lite Performance Benchmarks ===\n")
    
    # Generate test data
    print("Generating test data...")
    tables = {
        'users': pd.DataFrame({
            'user_id': range(args.benchmark_size),
            'username': [f'user_{i}' for i in range(args.benchmark_size)],
            'email': [f'user{i}@example.com' for i in range(args.benchmark_size)]
        }),
        'orders': pd.DataFrame({
            'order_id': range(args.benchmark_size * 2),
            'user_id': [i % args.benchmark_size for i in range(args.benchmark_size * 2)],
            'amount': [100.0 + i for i in range(args.benchmark_size * 2)]
        })
    }
    
    print(f"Created test tables with {args.benchmark_size} users and {args.benchmark_size * 2} orders")
    
    # Benchmark relationship analysis
    print("\nBenchmarking relationship analysis...")
    
    benchmark_result = benchmark_performance(
        export_schema_graph_detailed,
        tables,
        iterations=args.iterations
    )
    
    print(f"\nBenchmark Results:")
    print(f"  Mean Time: {benchmark_result['mean_time']:.3f}s")
    print(f"  Std Deviation: {benchmark_result['std_time']:.3f}s")
    print(f"  Min Time: {benchmark_result['min_time']:.3f}s")
    print(f"  Max Time: {benchmark_result['max_time']:.3f}s")
    print(f"  Total Time: {benchmark_result['total_time']:.3f}s")
    print(f"  Iterations: {benchmark_result['iterations']}")
    print(f"  Results Consistent: {benchmark_result['results_consistent']}")
    
    return 0


def test_command(args):
    """Run test suite."""
    print("=== Adel-Lite Enhanced Test Suite ===\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


def config_command(args):
    """Manage configuration."""
    if args.config_action == 'create':
        # Create default configuration file
        config = AdelLiteConfig()
        config.save_to_file(args.config_file or 'adel_config.json')
        print(f"Default configuration saved to {args.config_file or 'adel_config.json'}")
    
    elif args.config_action == 'show':
        # Show current configuration
        config = AdelLiteConfig()
        print("Current Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
    
    elif args.config_action == 'validate':
        # Validate configuration file
        if args.config_file:
            try:
                config = AdelLiteConfig.load_from_file(args.config_file)
                if config.validate():
                    print(f"✓ Configuration file {args.config_file} is valid")
                else:
                    print(f"✗ Configuration file {args.config_file} is invalid")
                    return 1
            except Exception as e:
                print(f"✗ Error loading configuration: {e}")
                return 1
        else:
            print("Error: No configuration file specified")
            return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adel-Lite Enhanced: Advanced database relationship detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic relationship analysis
  adel-lite-enhanced analyze file1.csv file2.csv -o results.json
  
  # Detailed analysis with configuration
  adel-lite-enhanced analyze file1.csv file2.csv --mode detailed -c config.json -o detailed_results.json
  
  # Run performance benchmarks
  adel-lite-enhanced benchmark --size 10000 --iterations 5
  
  # Run test suite
  adel-lite-enhanced test
  
  # Create default configuration
  adel-lite-enhanced config create -f my_config.json
        """
    )
    
    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze table relationships')
    analyze_parser.add_argument('csv_files', nargs='+', help='CSV files to analyze')
    analyze_parser.add_argument('-o', '--output', help='Output file path')
    analyze_parser.add_argument('-c', '--config', help='Configuration file path')
    analyze_parser.add_argument('--mode', choices=['basic', 'detailed'], default='detailed',
                               help='Analysis mode')
    analyze_parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                               help='Output format')
    analyze_parser.add_argument('--output-format', choices=['comprehensive', 'summary', 'relationships_only'],
                               default='comprehensive', help='Output detail level')
    analyze_parser.add_argument('--include-samples', action='store_true',
                               help='Include sample data in output')
    analyze_parser.add_argument('--sampling-strategy', choices=['random', 'stratified', 'systematic', 'adaptive'],
                               default='adaptive', help='Sampling strategy for large datasets')
    analyze_parser.add_argument('--summary-report', action='store_true',
                               help='Generate markdown summary report')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--size', dest='benchmark_size', type=int, default=1000,
                                 help='Test dataset size')
    benchmark_parser.add_argument('--iterations', type=int, default=3,
                                 help='Number of benchmark iterations')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('config_action', choices=['create', 'show', 'validate'],
                              help='Configuration action')
    config_parser.add_argument('-f', '--config-file', help='Configuration file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Execute command
    try:
        if args.command == 'analyze':
            return analyze_relationships(args)
        elif args.command == 'benchmark':
            return benchmark_command(args)
        elif args.command == 'test':
            return test_command(args)
        elif args.command == 'config':
            return config_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())