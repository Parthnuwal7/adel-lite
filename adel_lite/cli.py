"""
Command-line interface for adel-lite.
"""

import argparse
import sys
import glob
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from . import (
    schema, profile, map_relationships, detect_constraints,
    sample, visualize, export_schema, build_meta
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataframes_from_files(file_patterns: List[str]) -> tuple[List[pd.DataFrame], List[str]]:
    """
    Load DataFrames from file patterns.
    
    Args:
        file_patterns: List of file patterns (supports wildcards)
        
    Returns:
        Tuple of (dataframes_list, table_names_list)
    """
    df_list = []
    table_names = []
    
    for pattern in file_patterns:
        files = glob.glob(pattern)
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            continue
            
        for file_path in files:
            logger.info(f"Loading file: {file_path}")
            
            try:
                # Determine file type and load accordingly
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                df_list.append(df)
                # Use filename without extension as table name
                table_name = Path(file_path).stem
                table_names.append(table_name)
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
    
    return df_list, table_names


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adel-Lite: Automated Data Elements Linking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  adel-lite --input data/*.csv --output schema.json
  adel-lite --input customers.csv orders.csv --format yaml --output schema.yaml
  adel-lite --input data/*.csv --visualize --no-constraints
        """
    )
    
    # Input arguments
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='Input file patterns (supports wildcards)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        default='meta.json',
        help='Output file path (default: meta.json)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'yaml', 'ddl', 'avro'],
        default='json',
        help='Output format (default: json)'
    )
    
    # Analysis options
    parser.add_argument(
        '--no-constraints',
        action='store_true',
        help='Skip constraint detection'
    )
    
    parser.add_argument(
        '--constraint-threshold',
        type=float,
        default=0.95,
        help='Constraint detection threshold (default: 0.95)'
    )
    
    parser.add_argument(
        '--fk-threshold',
        type=float,
        default=0.8,
        help='Foreign key detection threshold (default: 0.8)'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate schema visualization'
    )
    
    parser.add_argument(
        '--viz-format',
        choices=['png', 'svg', 'pdf'],
        default='png',
        help='Visualization format (default: png)'
    )
    
    # Sampling options
    parser.add_argument(
        '--sample',
        type=int,
        metavar='N',
        help='Generate sample data (N rows per table)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output except errors'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    try:
        # Load data
        logger.info("Loading input files...")
        df_list, table_names = load_dataframes_from_files(args.input)
        
        if not df_list:
            logger.error("No valid input files found")
            sys.exit(1)
        
        logger.info(f"Loaded {len(df_list)} tables: {', '.join(table_names)}")
        
        # Run analysis
        logger.info("Generating schema...")
        schema_output = schema(df_list, table_names)
        
        logger.info("Generating profiles...")
        profile_output = profile(df_list, table_names)
        
        logger.info("Mapping relationships...")
        relationships_output = map_relationships(
            df_list, table_names, 
            fk_threshold=args.fk_threshold
        )
        
        # Optional constraint detection
        constraints_output = None
        if not args.no_constraints:
            logger.info("Detecting constraints...")
            constraints_output = detect_constraints(
                df_list, table_names,
                threshold=args.constraint_threshold
            )
        
        # Build final meta structure
        logger.info("Building meta structure...")
        meta = build_meta(
            schema_output, profile_output, 
            relationships_output, constraints_output
        )
        
        # Export results
        logger.info(f"Exporting to {args.format} format...")
        content = export_schema(meta, format=args.format, output_file=args.output)
        
        # Generate sample data if requested
        if args.sample:
            logger.info(f"Generating sample data ({args.sample} rows)...")
            sample_output = sample(df_list, table_names, n=args.sample)
            sample_file = args.output.replace('.', '_sample.')
            with open(sample_file, 'w') as f:
                json.dump(sample_output, f, indent=2, default=str)
            logger.info(f"Sample data saved to: {sample_file}")
        
        # Generate visualization if requested
        if args.visualize:
            logger.info("Generating visualization...")
            viz_file = args.output.replace('.', '_schema.')
            viz_path = visualize(
                schema_output, relationships_output,
                filename=viz_file.rsplit('.', 1)[0],
                format=args.viz_format
            )
            logger.info(f"Visualization saved to: {viz_path}")
        
        # Print summary
        if not args.quiet:
            from .meta import format_meta_summary
            print(format_meta_summary(meta))
        
        logger.info(f"Analysis complete! Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
