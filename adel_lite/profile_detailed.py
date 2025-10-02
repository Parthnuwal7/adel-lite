"""
Enhanced profile function with comprehensive column analysis and pattern detection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from .config import Config
from .pk_detection_detailed import analyze_column_quality, detect_pattern_type, calculate_column_entropy
from .scoring_detailed import calculate_entropy

logger = logging.getLogger(__name__)


def analyze_column_relationships(df: pd.DataFrame, column_name: str, 
                               sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze relationships and dependencies within a column.
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to analyze
        sample_size: Sample size for analysis (None for full data)
        
    Returns:
        Dictionary with relationship analysis
    """
    if sample_size is None:
        sample_size = Config.profile_sample_size
    
    series = df[column_name]
    
    # Sample data if needed
    if len(series) > sample_size:
        series = series.sample(n=sample_size, random_state=42)
    
    analysis = {
        'self_correlation': None,
        'autocorrelation_lag1': None,
        'trend_analysis': None,
        'seasonality_detected': False
    }
    
    # For numeric columns (but not boolean), calculate autocorrelation
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        try:
            # Remove nulls for correlation analysis
            clean_series = series.dropna()
            if len(clean_series) > 10:
                # Lag-1 autocorrelation
                if len(clean_series) > 1:
                    lag1_corr = clean_series.autocorr(lag=1)
                    analysis['autocorrelation_lag1'] = lag1_corr if not pd.isna(lag1_corr) else 0.0
                
                # Simple trend detection
                x = np.arange(len(clean_series))
                correlation = np.corrcoef(x, clean_series)[0, 1]
                analysis['trend_analysis'] = {
                    'trend_correlation': correlation if not np.isnan(correlation) else 0.0,
                    'has_trend': abs(correlation) > 0.3 if not np.isnan(correlation) else False
                }
        
        except Exception as e:
            logger.debug(f"Error in relationship analysis for {column_name}: {e}")
    
    return analysis


def calculate_column_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a column.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = {
        'count': len(series),
        'null_count': series.isnull().sum(),
        'unique_count': series.nunique(),
        'memory_usage': series.memory_usage(deep=True)
    }
    
    # Basic ratios
    stats['null_ratio'] = stats['null_count'] / stats['count'] if stats['count'] > 0 else 0.0
    stats['uniqueness_ratio'] = stats['unique_count'] / stats['count'] if stats['count'] > 0 else 0.0
    
    # Data type specific statistics
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        clean_series = series.dropna()
        if len(clean_series) > 0:
            stats.update({
                'min': float(clean_series.min()),
                'max': float(clean_series.max()),
                'mean': float(clean_series.mean()),
                'median': float(clean_series.median()),
                'std': float(clean_series.std()) if len(clean_series) > 1 else 0.0,
                'skewness': float(clean_series.skew()) if len(clean_series) > 1 else 0.0,
                'kurtosis': float(clean_series.kurtosis()) if len(clean_series) > 1 else 0.0,
                'q25': float(clean_series.quantile(0.25)),
                'q75': float(clean_series.quantile(0.75))
            })
            
            # Additional numeric insights
            stats['zero_count'] = (clean_series == 0).sum()
            stats['negative_count'] = (clean_series < 0).sum()
            stats['positive_count'] = (clean_series > 0).sum()
            stats['zero_ratio'] = stats['zero_count'] / len(clean_series)
    
    elif pd.api.types.is_bool_dtype(series):
        clean_series = series.dropna()
        if len(clean_series) > 0:
            true_count = clean_series.sum()
            false_count = len(clean_series) - true_count
            stats.update({
                'true_count': int(true_count),
                'false_count': int(false_count),
                'true_ratio': float(true_count) / len(clean_series),
                'false_ratio': float(false_count) / len(clean_series)
            })
    
    elif pd.api.types.is_string_dtype(series) or str(series.dtype) == 'object':
        clean_series = series.dropna().astype(str)
        if len(clean_series) > 0:
            str_lengths = clean_series.str.len()
            stats.update({
                'min_length': int(str_lengths.min()),
                'max_length': int(str_lengths.max()),
                'mean_length': float(str_lengths.mean()),
                'median_length': float(str_lengths.median()),
                'std_length': float(str_lengths.std()) if len(str_lengths) > 1 else 0.0
            })
            
            # String-specific insights
            stats['empty_string_count'] = (clean_series == '').sum()
            stats['whitespace_only_count'] = clean_series.str.strip().eq('').sum()
            stats['uppercase_count'] = clean_series.str.isupper().sum()
            stats['lowercase_count'] = clean_series.str.islower().sum()
            stats['numeric_string_count'] = clean_series.str.isnumeric().sum()
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        clean_series = series.dropna()
        if len(clean_series) > 0:
            stats.update({
                'min_date': clean_series.min(),
                'max_date': clean_series.max(),
                'date_range_days': (clean_series.max() - clean_series.min()).days
            })
    
    return stats


def detect_data_quality_issues(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Detect data quality issues in a column.
    
    Args:
        series: Pandas Series to analyze
        column_name: Name of the column
        
    Returns:
        Dictionary with data quality issues
    """
    issues = {
        'has_nulls': series.isnull().any(),
        'has_duplicates': series.duplicated().any(),
        'has_outliers': False,
        'has_inconsistent_format': False,
        'potential_issues': []
    }
    
    # Check for high null ratio
    null_ratio = series.isnull().sum() / len(series)
    if null_ratio > 0.5:
        issues['potential_issues'].append(f"High null ratio: {null_ratio:.2f}")
    
    # For numeric columns (but not boolean), check for outliers using IQR method
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        clean_series = series.dropna()
        if len(clean_series) > 4:
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            
            if len(outliers) > 0:
                issues['has_outliers'] = True
                outlier_ratio = len(outliers) / len(clean_series)
                issues['potential_issues'].append(f"Outliers detected: {len(outliers)} ({outlier_ratio:.2%})")
    
    # For string columns, check format consistency
    elif pd.api.types.is_string_dtype(series) or str(series.dtype) == 'object':
        clean_series = series.dropna().astype(str)
        if len(clean_series) > 1:
            # Check length variation
            str_lengths = clean_series.str.len()
            length_std = str_lengths.std()
            length_mean = str_lengths.mean()
            
            if length_std > length_mean * 0.5:  # High variation in string lengths
                issues['has_inconsistent_format'] = True
                issues['potential_issues'].append(f"Inconsistent string lengths (std: {length_std:.1f})")
            
            # Check for mixed case patterns
            has_upper = clean_series.str.contains(r'[A-Z]', na=False).any()
            has_lower = clean_series.str.contains(r'[a-z]', na=False).any()
            
            if has_upper and has_lower:
                upper_count = clean_series.str.isupper().sum()
                lower_count = clean_series.str.islower().sum()
                mixed_count = len(clean_series) - upper_count - lower_count
                
                if mixed_count > len(clean_series) * 0.3:
                    issues['potential_issues'].append("Mixed case patterns detected")
    
    # Check for extremely low or high cardinality
    cardinality_ratio = series.nunique() / len(series)
    if cardinality_ratio < 0.01 and series.nunique() > 1:
        issues['potential_issues'].append(f"Very low cardinality: {series.nunique()} unique values")
    elif cardinality_ratio > 0.95 and len(series) > 100:
        issues['potential_issues'].append(f"Very high cardinality: {cardinality_ratio:.2%} unique")
    
    return issues


def generate_column_insights(analysis: Dict[str, Any]) -> List[str]:
    """
    Generate human-readable insights from column analysis.
    
    Args:
        analysis: Column analysis results
        
    Returns:
        List of insight strings
    """
    insights = []
    
    # Uniqueness insights
    uniqueness = analysis['statistics']['uniqueness_ratio']
    if uniqueness == 1.0:
        insights.append("Perfect uniqueness - excellent primary key candidate")
    elif uniqueness > 0.95:
        insights.append("Very high uniqueness - good primary key candidate")
    elif uniqueness < 0.1:
        insights.append("Low uniqueness - categorical or reference data")
    
    # Null insights
    null_ratio = analysis['statistics']['null_ratio']
    if null_ratio == 0.0:
        insights.append("No missing values - complete data")
    elif null_ratio > 0.5:
        insights.append("High missing data rate - data quality concern")
    
    # Pattern insights
    pattern_type = analysis['pattern_type']
    pattern_confidence = analysis['pattern_confidence']
    
    if pattern_type != 'unknown' and pattern_confidence > 0.8:
        insights.append(f"Strong {pattern_type} pattern detected ({pattern_confidence:.1%} confidence)")
    
    # Entropy insights
    entropy = analysis['entropy']
    if entropy > 10:
        insights.append("Very high entropy - highly diverse data")
    elif entropy < 2:
        insights.append("Low entropy - limited diversity")
    
    # Data type specific insights
    if analysis['statistics'].get('is_numeric', False):
        if 'zero_ratio' in analysis['statistics'] and analysis['statistics']['zero_ratio'] > 0.3:
            insights.append("High proportion of zero values")
        
        if 'skewness' in analysis['statistics']:
            skew = analysis['statistics']['skewness']
            if abs(skew) > 2:
                direction = "right" if skew > 0 else "left"
                insights.append(f"Highly {direction}-skewed distribution")
    
    # Quality issues
    if analysis['data_quality']['has_outliers']:
        insights.append("Outliers detected - may need data cleaning")
    
    if analysis['data_quality']['has_inconsistent_format']:
        insights.append("Inconsistent formatting detected")
    
    return insights


def profile_detailed(df: pd.DataFrame, table_name: Optional[str] = None,
                    include_relationships: bool = True,
                    include_quality_check: bool = True,
                    sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate comprehensive profile of a DataFrame with enhanced analysis.
    
    Args:
        df: DataFrame to profile
        table_name: Name of the table (optional)
        include_relationships: Whether to include relationship analysis
        include_quality_check: Whether to include quality checks
        sample_size: Sample size for analysis (None for auto-sizing)
        
    Returns:
        Comprehensive profile dictionary
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for table {table_name or 'unnamed'}")
        return {'error': 'Empty DataFrame', 'table': table_name}
    
    if table_name is None:
        table_name = 'unnamed_table'
    
    if sample_size is None:
        sample_size = min(Config.profile_sample_size, len(df))
    
    logger.info(f"Profiling table {table_name} with {len(df)} rows and {len(df.columns)} columns")
    
    # Overall table statistics
    table_stats = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'total_cells': len(df) * len(df.columns),
        'null_cells': df.isnull().sum().sum(),
        'non_null_cells': df.count().sum()
    }
    
    table_stats['null_cell_ratio'] = (table_stats['null_cells'] / 
                                     table_stats['total_cells'] if table_stats['total_cells'] > 0 else 0.0)
    
    # Column-by-column analysis
    column_profiles = {}
    
    for column_name in df.columns:
        try:
            logger.debug(f"Profiling column {column_name}")
            
            series = df[column_name]
            
            # Basic quality analysis
            basic_analysis = analyze_column_quality(series, column_name)
            
            # Enhanced statistics
            statistics = calculate_column_statistics(series)
            
            # Relationship analysis
            relationships = {}
            if include_relationships:
                relationships = analyze_column_relationships(df, column_name, sample_size)
            
            # Data quality issues
            data_quality = {}
            if include_quality_check:
                data_quality = detect_data_quality_issues(series, column_name)
            
            # Combine all analysis
            column_profile = {
                'column_name': column_name,
                'statistics': statistics,
                'entropy': basic_analysis['entropy'],
                'pattern_type': basic_analysis['pattern_type'],
                'pattern_confidence': basic_analysis['pattern_confidence'],
                'pattern_metadata': basic_analysis['pattern_metadata'],
                'relationships': relationships,
                'data_quality': data_quality,
                'sample_values': basic_analysis['sample_values'][:5],  # Limit sample size
                'insights': []
            }
            
            # Generate insights
            column_profile['insights'] = generate_column_insights(column_profile)
            
            column_profiles[column_name] = column_profile
        
        except Exception as e:
            logger.error(f"Error profiling column {column_name}: {e}")
            column_profiles[column_name] = {
                'column_name': column_name,
                'error': str(e),
                'statistics': {'count': len(df[column_name])}
            }
    
    # Table-level insights
    table_insights = []
    
    # Identify potential primary keys
    pk_candidates = []
    for col_name, profile in column_profiles.items():
        if 'statistics' in profile and profile['statistics'].get('uniqueness_ratio', 0) > 0.95:
            pk_candidates.append(col_name)
    
    if pk_candidates:
        table_insights.append(f"Potential primary key columns: {', '.join(pk_candidates)}")
    else:
        table_insights.append("No obvious primary key candidates found")
    
    # Data quality summary
    columns_with_nulls = sum(1 for profile in column_profiles.values() 
                           if profile.get('statistics', {}).get('null_ratio', 0) > 0)
    
    if columns_with_nulls > 0:
        table_insights.append(f"{columns_with_nulls} columns have missing values")
    
    # Pattern distribution
    pattern_counts = {}
    for profile in column_profiles.values():
        pattern = profile.get('pattern_type', 'unknown')
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    if pattern_counts.get('uuid', 0) > 0:
        table_insights.append(f"{pattern_counts['uuid']} UUID columns detected")
    
    if pattern_counts.get('email', 0) > 0:
        table_insights.append(f"{pattern_counts['email']} email columns detected")
    
    # Compile final profile
    profile_result = {
        'table_name': table_name,
        'table_statistics': table_stats,
        'table_insights': table_insights,
        'column_profiles': column_profiles,
        'column_count': len(column_profiles),
        'successful_profiles': len([p for p in column_profiles.values() if 'error' not in p]),
        'failed_profiles': len([p for p in column_profiles.values() if 'error' in p]),
        'pattern_distribution': pattern_counts,
        'metadata': {
            'profile_timestamp': pd.Timestamp.now().isoformat(),
            'sample_size_used': sample_size,
            'include_relationships': include_relationships,
            'include_quality_check': include_quality_check,
            'config_version': Config.version
        }
    }
    
    logger.info(f"Completed profiling for table {table_name}: "
                f"{profile_result['successful_profiles']}/{len(df.columns)} columns successful")
    
    return profile_result


def compare_profiles(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two table profiles to identify differences and similarities.
    
    Args:
        profile1: First table profile
        profile2: Second table profile
        
    Returns:
        Comparison analysis
    """
    comparison = {
        'table1_name': profile1.get('table_name', 'table1'),
        'table2_name': profile2.get('table_name', 'table2'),
        'comparison_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Compare table-level statistics
    stats1 = profile1.get('table_statistics', {})
    stats2 = profile2.get('table_statistics', {})
    
    comparison['table_comparison'] = {
        'row_count_diff': stats2.get('row_count', 0) - stats1.get('row_count', 0),
        'column_count_diff': stats2.get('column_count', 0) - stats1.get('column_count', 0),
        'memory_usage_diff_mb': stats2.get('memory_usage_mb', 0) - stats1.get('memory_usage_mb', 0),
        'null_ratio_diff': stats2.get('null_cell_ratio', 0) - stats1.get('null_cell_ratio', 0)
    }
    
    # Compare columns
    cols1 = set(profile1.get('column_profiles', {}).keys())
    cols2 = set(profile2.get('column_profiles', {}).keys())
    
    comparison['column_comparison'] = {
        'common_columns': list(cols1 & cols2),
        'only_in_table1': list(cols1 - cols2),
        'only_in_table2': list(cols2 - cols1),
        'total_unique_columns': len(cols1 | cols2)
    }
    
    # Compare common columns in detail
    common_column_analysis = {}
    for col in comparison['column_comparison']['common_columns']:
        col_profile1 = profile1['column_profiles'][col]
        col_profile2 = profile2['column_profiles'][col]
        
        if 'statistics' in col_profile1 and 'statistics' in col_profile2:
            stats_diff = {}
            stats1_col = col_profile1['statistics']
            stats2_col = col_profile2['statistics']
            
            for stat_name in ['uniqueness_ratio', 'null_ratio', 'count']:
                if stat_name in stats1_col and stat_name in stats2_col:
                    stats_diff[f'{stat_name}_diff'] = stats2_col[stat_name] - stats1_col[stat_name]
            
            common_column_analysis[col] = {
                'statistics_diff': stats_diff,
                'pattern_change': {
                    'from': col_profile1.get('pattern_type', 'unknown'),
                    'to': col_profile2.get('pattern_type', 'unknown'),
                    'changed': col_profile1.get('pattern_type') != col_profile2.get('pattern_type')
                }
            }
    
    comparison['common_column_analysis'] = common_column_analysis
    
    # Pattern distribution comparison
    patterns1 = profile1.get('pattern_distribution', {})
    patterns2 = profile2.get('pattern_distribution', {})
    
    all_patterns = set(patterns1.keys()) | set(patterns2.keys())
    pattern_comparison = {}
    
    for pattern in all_patterns:
        count1 = patterns1.get(pattern, 0)
        count2 = patterns2.get(pattern, 0)
        pattern_comparison[pattern] = {
            'table1_count': count1,
            'table2_count': count2,
            'difference': count2 - count1
        }
    
    comparison['pattern_distribution_comparison'] = pattern_comparison
    
    return comparison


def export_profile_summary(profile: Dict[str, Any], format_type: str = 'dict') -> Union[Dict[str, Any], str]:
    """
    Export a simplified summary of the profile.
    
    Args:
        profile: Profile result to summarize
        format_type: Output format ('dict' or 'markdown')
        
    Returns:
        Summary in requested format
    """
    if format_type == 'dict':
        summary = {
            'table': profile.get('table_name'),
            'rows': profile.get('table_statistics', {}).get('row_count'),
            'columns': profile.get('column_count'),
            'memory_mb': round(profile.get('table_statistics', {}).get('memory_usage_mb', 0), 2),
            'null_ratio': round(profile.get('table_statistics', {}).get('null_cell_ratio', 0), 3),
            'column_summary': {}
        }
        
        for col_name, col_profile in profile.get('column_profiles', {}).items():
            if 'error' not in col_profile:
                summary['column_summary'][col_name] = {
                    'type': col_profile.get('statistics', {}).get('dtype'),
                    'pattern': col_profile.get('pattern_type'),
                    'uniqueness': round(col_profile.get('statistics', {}).get('uniqueness_ratio', 0), 3),
                    'nulls': round(col_profile.get('statistics', {}).get('null_ratio', 0), 3),
                    'entropy': round(col_profile.get('entropy', 0), 2)
                }
        
        return summary
    
    elif format_type == 'markdown':
        md_lines = [
            f"# Profile Summary: {profile.get('table_name', 'Unknown')}",
            "",
            "## Table Statistics",
            f"- **Rows:** {profile.get('table_statistics', {}).get('row_count', 'N/A'):,}",
            f"- **Columns:** {profile.get('column_count', 'N/A')}",
            f"- **Memory Usage:** {profile.get('table_statistics', {}).get('memory_usage_mb', 0):.2f} MB",
            f"- **Null Cell Ratio:** {profile.get('table_statistics', {}).get('null_cell_ratio', 0):.1%}",
            "",
            "## Column Analysis",
            "",
            "| Column | Type | Pattern | Uniqueness | Nulls | Entropy |",
            "|--------|------|---------|------------|-------|---------|"
        ]
        
        for col_name, col_profile in profile.get('column_profiles', {}).items():
            if 'error' not in col_profile:
                stats = col_profile.get('statistics', {})
                dtype = stats.get('dtype', 'unknown')
                pattern = col_profile.get('pattern_type', 'unknown')
                uniqueness = stats.get('uniqueness_ratio', 0)
                nulls = stats.get('null_ratio', 0)
                entropy = col_profile.get('entropy', 0)
                
                md_lines.append(
                    f"| {col_name} | {dtype} | {pattern} | {uniqueness:.1%} | {nulls:.1%} | {entropy:.1f} |"
                )
        
        return "\n".join(md_lines)
    
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")