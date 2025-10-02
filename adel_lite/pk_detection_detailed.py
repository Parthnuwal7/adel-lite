"""
Enhanced primary key detection with pattern recognition and configurable thresholds.
"""

import pandas as pd
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from .config import Config
from .scoring_detailed import (
    calculate_entropy, 
    check_monotonic_sequence, 
    score_primary_key_candidate
)

logger = logging.getLogger(__name__)


def detect_pattern_type(values: List[Any], sample_size: int = 100) -> Tuple[str, float, Dict[str, Any]]:
    """
    Detect the pattern type of values (UUID, email, int_sequence, etc.).
    
    Args:
        values: List of values to analyze
        sample_size: Number of values to sample for pattern detection
        
    Returns:
        Tuple of (pattern_type, confidence, metadata)
    """
    if not values:
        return 'unknown', 0.0, {}
    
    # Remove nulls and sample
    non_null_values = [v for v in values if v is not None]
    if not non_null_values:
        return 'unknown', 0.0, {}
    
    if len(non_null_values) > sample_size:
        import random
        non_null_values = random.sample(non_null_values, sample_size)
    
    # Convert to strings for pattern matching
    str_values = [str(v) for v in non_null_values]
    total_count = len(str_values)
    
    # Pattern detection results
    pattern_results = {}
    
    # UUID pattern detection
    uuid_pattern = re.compile(Config.pattern_configs['uuid']['regex'], re.IGNORECASE)
    uuid_matches = sum(1 for v in str_values if uuid_pattern.match(v))
    uuid_confidence = uuid_matches / total_count if total_count > 0 else 0.0
    pattern_results['uuid'] = {
        'matches': uuid_matches,
        'confidence': uuid_confidence,
        'threshold': Config.pattern_configs['uuid']['confidence_threshold']
    }
    
    # Email pattern detection
    email_pattern = re.compile(Config.pattern_configs['email']['regex'], re.IGNORECASE)
    email_matches = sum(1 for v in str_values if email_pattern.match(v))
    email_confidence = email_matches / total_count if total_count > 0 else 0.0
    pattern_results['email'] = {
        'matches': email_matches,
        'confidence': email_confidence,
        'threshold': Config.pattern_configs['email']['confidence_threshold']
    }
    
    # Phone pattern detection
    phone_pattern = re.compile(Config.pattern_configs['phone']['regex'])
    phone_matches = sum(1 for v in str_values if phone_pattern.match(v.replace('-', '').replace(' ', '')))
    phone_confidence = phone_matches / total_count if total_count > 0 else 0.0
    pattern_results['phone'] = {
        'matches': phone_matches,
        'confidence': phone_confidence,
        'threshold': Config.pattern_configs['phone']['confidence_threshold']
    }
    
    # URL pattern detection
    url_pattern = re.compile(Config.pattern_configs['url']['regex'], re.IGNORECASE)
    url_matches = sum(1 for v in str_values if url_pattern.match(v))
    url_confidence = url_matches / total_count if total_count > 0 else 0.0
    pattern_results['url'] = {
        'matches': url_matches,
        'confidence': url_confidence,
        'threshold': Config.pattern_configs['url']['confidence_threshold']
    }
    
    # Integer sequence detection
    int_sequence_confidence = 0.0
    is_monotonic = False
    monotonic_ratio = 0.0
    
    try:
        # Check if values can be converted to integers
        numeric_values = []
        for v in non_null_values:
            try:
                numeric_values.append(int(float(v)))
            except (ValueError, TypeError):
                break
        
        if len(numeric_values) == len(non_null_values) and len(numeric_values) >= Config.pattern_configs['int_sequence']['min_samples']:
            is_monotonic, monotonic_ratio = check_monotonic_sequence(numeric_values, 
                                                                   Config.pattern_configs['int_sequence']['gap_tolerance'])
            
            if monotonic_ratio >= Config.pattern_configs['int_sequence']['monotonic_threshold']:
                int_sequence_confidence = monotonic_ratio
    
    except Exception as e:
        logger.debug(f"Error detecting integer sequence: {e}")
    
    pattern_results['int_sequence'] = {
        'is_monotonic': is_monotonic,
        'monotonic_ratio': monotonic_ratio,
        'confidence': int_sequence_confidence,
        'threshold': Config.pattern_configs['int_sequence']['monotonic_threshold']
    }
    
    # Determine the best pattern match
    best_pattern = 'unknown'
    best_confidence = 0.0
    best_metadata = {}
    
    for pattern_type, result in pattern_results.items():
        confidence = result['confidence']
        threshold = result.get('threshold', 0.5)
        
        if confidence >= threshold and confidence > best_confidence:
            best_pattern = pattern_type
            best_confidence = confidence
            best_metadata = result
    
    return best_pattern, best_confidence, pattern_results


def calculate_column_entropy(series: pd.Series, max_samples: Optional[int] = None) -> float:
    """
    Calculate entropy for a pandas Series.
    
    Args:
        series: Pandas Series to analyze
        max_samples: Maximum samples for entropy calculation
        
    Returns:
        Shannon entropy value
    """
    if max_samples is None:
        max_samples = Config.entropy_sample_size
    
    non_null_values = series.dropna().tolist()
    return calculate_entropy(non_null_values, max_samples)


def analyze_column_quality(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Perform comprehensive quality analysis of a column for PK candidacy.
    
    Args:
        series: Pandas Series to analyze
        column_name: Name of the column
        
    Returns:
        Dictionary with quality metrics
    """
    total_count = len(series)
    null_count = series.isnull().sum()
    unique_count = series.nunique()
    
    # Basic ratios
    null_ratio = null_count / total_count if total_count > 0 else 0.0
    uniqueness_ratio = unique_count / total_count if total_count > 0 else 0.0
    
    # Entropy calculation
    entropy = calculate_column_entropy(series)
    
    # Pattern detection
    non_null_values = series.dropna().tolist()
    pattern_type, pattern_confidence, pattern_metadata = detect_pattern_type(non_null_values)
    
    # Monotonic sequence check for numeric data
    monotonic_ratio = None
    if pattern_type == 'int_sequence':
        monotonic_ratio = pattern_metadata.get('monotonic_ratio', 0.0)
    
    # Data type analysis
    dtype_str = str(series.dtype)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_string = pd.api.types.is_string_dtype(series) or dtype_str == 'object'
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    
    # Value statistics
    value_stats = {}
    if is_numeric and unique_count > 0:
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            value_stats.update({
                'min': float(non_null_series.min()),
                'max': float(non_null_series.max()),
                'mean': float(non_null_series.mean()),
                'std': float(non_null_series.std()) if len(non_null_series) > 1 else 0.0
            })
    
    if is_string and unique_count > 0:
        str_lengths = series.dropna().astype(str).str.len()
        if len(str_lengths) > 0:
            value_stats.update({
                'min_length': int(str_lengths.min()),
                'max_length': int(str_lengths.max()),
                'avg_length': float(str_lengths.mean()),
                'std_length': float(str_lengths.std()) if len(str_lengths) > 1 else 0.0
            })
    
    # Cardinality analysis
    cardinality_ratio = unique_count / total_count if total_count > 0 else 0.0
    
    # Sample values for inspection
    sample_values = series.dropna().head(10).tolist()
    
    return {
        'column_name': column_name,
        'total_count': total_count,
        'null_count': null_count,
        'unique_count': unique_count,
        'null_ratio': null_ratio,
        'uniqueness_ratio': uniqueness_ratio,
        'cardinality_ratio': cardinality_ratio,
        'entropy': entropy,
        'pattern_type': pattern_type,
        'pattern_confidence': pattern_confidence,
        'pattern_metadata': pattern_metadata,
        'monotonic_ratio': monotonic_ratio,
        'dtype': dtype_str,
        'is_numeric': is_numeric,
        'is_string': is_string,
        'is_datetime': is_datetime,
        'value_stats': value_stats,
        'sample_values': sample_values
    }


def is_primary_key_candidate(column_analysis: Dict[str, Any], 
                           strict_mode: Optional[bool] = None) -> Tuple[bool, str]:
    """
    Determine if a column is a primary key candidate based on analysis.
    
    Args:
        column_analysis: Column analysis results
        strict_mode: Use strict thresholds (uses config default if None)
        
    Returns:
        Tuple of (is_candidate, reason)
    """
    if strict_mode is None:
        strict_mode = Config.strict_mode
    
    # Get thresholds based on mode
    if strict_mode:
        uniqueness_threshold = 1.0
        null_threshold = 0.0
        entropy_threshold = Config.pk_entropy_threshold * 1.5
    else:
        uniqueness_threshold = Config.pk_uniqueness_threshold
        null_threshold = Config.pk_null_tolerance
        entropy_threshold = Config.pk_entropy_threshold
    
    # Check basic requirements
    uniqueness_ratio = column_analysis['uniqueness_ratio']
    null_ratio = column_analysis['null_ratio']
    entropy = column_analysis['entropy']
    unique_count = column_analysis['unique_count']
    
    # Uniqueness check
    if uniqueness_ratio < uniqueness_threshold:
        return False, f"Uniqueness {uniqueness_ratio:.3f} < {uniqueness_threshold}"
    
    # Null check
    if null_ratio > null_threshold:
        return False, f"Null ratio {null_ratio:.3f} > {null_threshold}"
    
    # Entropy check
    if entropy < entropy_threshold:
        return False, f"Entropy {entropy:.2f} < {entropy_threshold}"
    
    # Minimum cardinality check
    if unique_count < Config.pk_min_cardinality:
        return False, f"Cardinality {unique_count} < {Config.pk_min_cardinality}"
    
    # Check for datetime columns (usually not good PKs)
    if column_analysis['is_datetime'] and strict_mode:
        return False, "Datetime columns excluded in strict mode"
    
    # All checks passed
    reasons = []
    if uniqueness_ratio == 1.0:
        reasons.append("perfect uniqueness")
    if null_ratio == 0.0:
        reasons.append("no nulls")
    if column_analysis['pattern_type'] in ['uuid', 'int_sequence']:
        reasons.append(f"{column_analysis['pattern_type']} pattern")
    
    return True, f"Passed all checks: {', '.join(reasons)}"


def detect_primary_keys_detailed(df: pd.DataFrame, table_name: str) -> List[Dict[str, Any]]:
    """
    Detect primary key candidates with detailed analysis and scoring.
    
    Args:
        df: DataFrame to analyze
        table_name: Name of the table
        
    Returns:
        List of primary key candidates with detailed metadata
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for table {table_name}")
        return []
    
    pk_candidates = []
    
    logger.info(f"Analyzing {len(df.columns)} columns in table {table_name} for PK candidates")
    
    for column_name in df.columns:
        try:
            series = df[column_name]
            
            # Perform comprehensive analysis
            analysis = analyze_column_quality(series, column_name)
            
            # Check if it's a PK candidate
            is_candidate, reason = is_primary_key_candidate(analysis)
            
            # Calculate score using detailed scoring
            scoring_result = score_primary_key_candidate(analysis)
            
            # Create candidate record
            candidate = {
                'table': table_name,
                'column': column_name,
                'is_candidate': is_candidate,
                'reason': reason,
                'score': scoring_result['score'],
                'decision': scoring_result['decision'],
                'explanation': scoring_result['explanation'],
                'detailed_explanation': scoring_result['detailed_explanation'],
                'analysis': analysis,
                'features': scoring_result['features'],
                'metadata': {
                    'uniqueness_ratio': analysis['uniqueness_ratio'],
                    'null_ratio': analysis['null_ratio'],
                    'entropy': analysis['entropy'],
                    'pattern_type': analysis['pattern_type'],
                    'pattern_confidence': analysis['pattern_confidence'],
                    'cardinality': analysis['unique_count'],
                    'total_rows': analysis['total_count'],
                    'dtype': analysis['dtype']
                }
            }
            
            pk_candidates.append(candidate)
            
            # Log results for significant candidates
            if is_candidate or scoring_result['score'] > 0.5:
                logger.info(f"PK candidate {table_name}.{column_name}: "
                          f"score={scoring_result['score']:.3f}, "
                          f"decision={scoring_result['decision']}, "
                          f"pattern={analysis['pattern_type']}")
        
        except Exception as e:
            logger.error(f"Error analyzing column {column_name} in table {table_name}: {e}")
            continue
    
    # Sort by score (highest first)
    pk_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Log summary
    accepted_candidates = [c for c in pk_candidates if c['decision'] == 'accepted']
    logger.info(f"Found {len(accepted_candidates)} accepted PK candidates in table {table_name}")
    
    return pk_candidates


def get_best_primary_key_candidate(pk_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Get the best primary key candidate from a list.
    
    Args:
        pk_candidates: List of PK candidates
        
    Returns:
        Best candidate or None if no good candidates
    """
    if not pk_candidates:
        return None
    
    # Filter to accepted candidates first
    accepted = [c for c in pk_candidates if c['decision'] == 'accepted']
    
    if accepted:
        # Return highest scoring accepted candidate
        return max(accepted, key=lambda x: x['score'])
    
    # If no accepted candidates, return highest scoring overall if above threshold
    best = max(pk_candidates, key=lambda x: x['score'])
    if best['score'] > 0.5:  # Reasonable threshold
        return best
    
    return None


def compare_primary_key_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple primary key candidates and provide analysis.
    
    Args:
        candidates: List of PK candidates to compare
        
    Returns:
        Comparison analysis
    """
    if not candidates:
        return {'comparison': 'No candidates to compare'}
    
    # Sort by score
    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    best = sorted_candidates[0]
    
    comparison = {
        'total_candidates': len(candidates),
        'accepted_count': len([c for c in candidates if c['decision'] == 'accepted']),
        'rejected_count': len([c for c in candidates if c['decision'] == 'rejected']),
        'ambiguous_count': len([c for c in candidates if c['decision'] == 'ambiguous']),
        'best_candidate': {
            'column': best['column'],
            'score': best['score'],
            'decision': best['decision'],
            'pattern_type': best['analysis']['pattern_type']
        },
        'score_distribution': {
            'max': max(c['score'] for c in candidates),
            'min': min(c['score'] for c in candidates),
            'avg': sum(c['score'] for c in candidates) / len(candidates)
        }
    }
    
    # Add pattern type distribution
    pattern_counts = {}
    for candidate in candidates:
        pattern = candidate['analysis']['pattern_type']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    comparison['pattern_distribution'] = pattern_counts
    
    # Add recommendations
    recommendations = []
    
    if comparison['accepted_count'] == 0:
        recommendations.append("No strong PK candidates found. Consider composite keys.")
    elif comparison['accepted_count'] == 1:
        recommendations.append(f"Clear PK choice: {best['column']}")
    else:
        recommendations.append(f"Multiple PK candidates. Recommend: {best['column']} (highest score)")
    
    if any(c['analysis']['pattern_type'] == 'uuid' for c in candidates):
        recommendations.append("UUID column(s) present - excellent PK choice")
    
    if any(c['analysis']['pattern_type'] == 'int_sequence' for c in candidates):
        recommendations.append("Sequential integer column(s) present - good surrogate key")
    
    comparison['recommendations'] = recommendations
    
    return comparison