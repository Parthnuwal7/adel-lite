"""
Detailed scoring functions for Adel-Lite primary key and foreign key detection.
Implements weighted scoring with sigmoid normalization and decision thresholds.
"""

import numpy as np
import math
from typing import Dict, Any, Tuple, Optional, List
import logging
from .config import Config

logger = logging.getLogger(__name__)


def sigmoid(x: float, steepness: float = 10.0, midpoint: float = 0.5) -> float:
    """
    Apply sigmoid function for smooth [0,1] normalization.
    
    Args:
        x: Input value to normalize
        steepness: Controls the steepness of the sigmoid curve
        midpoint: Point where sigmoid equals 0.5
        
    Returns:
        Normalized value in [0, 1]
    """
    try:
        return 1 / (1 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 1.0 if x > midpoint else 0.0


def calculate_entropy(values: List[Any], max_samples: int = 1000) -> float:
    """
    Calculate Shannon entropy of a list of values.
    
    Args:
        values: List of values to analyze
        max_samples: Maximum number of samples for performance
        
    Returns:
        Entropy value (higher = more diverse)
    """
    if not values or len(values) == 0:
        return 0.0
    
    # Sample if too many values
    if len(values) > max_samples:
        import random
        values = random.sample(values, max_samples)
    
    # Count frequencies
    freq_count = {}
    for value in values:
        str_val = str(value)  # Convert to string for consistency
        freq_count[str_val] = freq_count.get(str_val, 0) + 1
    
    # Calculate entropy
    total = len(values)
    entropy = 0.0
    
    for count in freq_count.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def check_monotonic_sequence(values: List[Any], tolerance: float = 0.1) -> Tuple[bool, float]:
    """
    Check if values form a monotonic sequence (typical of surrogate keys).
    
    Args:
        values: List of numeric values
        tolerance: Fraction of gaps allowed
        
    Returns:
        Tuple of (is_monotonic, monotonic_ratio)
    """
    try:
        # Convert to numeric and sort
        numeric_values = []
        for val in values:
            if val is not None:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    return False, 0.0
        
        if len(numeric_values) < 2:
            return False, 0.0
        
        # Sort values and check for monotonic pattern
        sorted_vals = sorted(numeric_values)
        
        # Check if values are mostly sequential
        gaps = 0
        expected_gaps = 0
        
        for i in range(1, len(sorted_vals)):
            diff = sorted_vals[i] - sorted_vals[i-1]
            if diff > 1.1:  # Allow small floating point errors
                gaps += 1
            expected_gaps += 1
        
        gap_ratio = gaps / expected_gaps if expected_gaps > 0 else 0.0
        is_monotonic = gap_ratio <= tolerance
        monotonic_ratio = 1.0 - gap_ratio
        
        return is_monotonic, monotonic_ratio
        
    except Exception as e:
        logger.debug(f"Error checking monotonic sequence: {e}")
        return False, 0.0


def calculate_pattern_match_score(fk_values: List[Any], pk_values: List[Any]) -> float:
    """
    Calculate pattern matching score between FK and PK values.
    
    Args:
        fk_values: Foreign key values
        pk_values: Primary key values
        
    Returns:
        Pattern match score [0, 1]
    """
    try:
        # Sample values for pattern analysis
        sample_size = min(100, len(fk_values), len(pk_values))
        if sample_size < 2:
            return 0.0
        
        import random
        fk_sample = random.sample(fk_values, min(sample_size, len(fk_values)))
        pk_sample = random.sample(pk_values, min(sample_size, len(pk_values)))
        
        # Check data type consistency
        fk_types = set(type(v).__name__ for v in fk_sample if v is not None)
        pk_types = set(type(v).__name__ for v in pk_sample if v is not None)
        
        type_overlap = len(fk_types.intersection(pk_types)) / max(len(fk_types.union(pk_types)), 1)
        
        # Check string pattern similarity (if string data)
        pattern_score = 0.0
        if all(isinstance(v, str) for v in fk_sample[:10]) and all(isinstance(v, str) for v in pk_sample[:10]):
            # Compare string length distributions
            fk_lengths = [len(str(v)) for v in fk_sample if v is not None]
            pk_lengths = [len(str(v)) for v in pk_sample if v is not None]
            
            if fk_lengths and pk_lengths:
                fk_avg_len = np.mean(fk_lengths)
                pk_avg_len = np.mean(pk_lengths)
                length_similarity = 1.0 - abs(fk_avg_len - pk_avg_len) / max(fk_avg_len, pk_avg_len, 1)
                pattern_score = length_similarity
        
        # Combine type overlap and pattern similarity
        return (type_overlap + pattern_score) / 2.0
        
    except Exception as e:
        logger.debug(f"Error calculating pattern match: {e}")
        return 0.0


def calculate_fk_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate weighted foreign key relationship score from features dictionary.
    
    Args:
        features: Dictionary containing FK relationship features
        
    Returns:
        Dictionary with score, decision, explanation, and detailed_explanation
    """
    # Extract features with defaults
    coverage_fk_to_pk = features.get('fk_to_pk_coverage', 0.0)
    coverage_pk_to_fk = features.get('pk_to_fk_coverage', 0.0)
    fk_null_ratio = features.get('fk_null_ratio', 1.0)
    constraint_violation_ratio = features.get('constraint_violation_ratio', 1.0)
    dtype_compatibility_score = features.get('dtype_compatibility_score', 0.0)
    pattern_consistency_score = features.get('pattern_consistency_score', 0.0)
    name_similarity = features.get('name_similarity', 0.0)
    
    return calculate_fk_score_detailed(
        coverage_fk_to_pk=coverage_fk_to_pk,
        coverage_pk_to_fk=coverage_pk_to_fk,
        pk_uniqueness=1.0,  # Assume PK is unique
        fk_null_ratio=fk_null_ratio,
        datatype_match=dtype_compatibility_score,
        pattern_match=pattern_consistency_score,
        name_fuzzy_similarity=name_similarity,
        constraint_violation_rate=constraint_violation_ratio
    )


def calculate_pk_score_detailed(
    uniqueness_ratio: float,
    null_ratio: float,
    entropy_score: float,
    pattern_bonus: float = 0.0,
    sequence_penalty: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate weighted primary key score.
    
    Args:
        uniqueness_ratio: Ratio of unique values (0-1)
        null_ratio: Ratio of null values (0-1)
        entropy_score: Entropy score (0-1)
        pattern_bonus: Bonus for recognized patterns (0-1)
        sequence_penalty: Penalty for sequential patterns (0-1)
        weights: Custom weights (uses config defaults if None)
        
    Returns:
        Primary key score result dictionary
    """
    if weights is None:
        weights = {
            'uniqueness': 0.4,
            'completeness': 0.3,
            'entropy': 0.2,
            'pattern_bonus': 0.1,
            'sequence_penalty': -0.1
        }
    
    # Calculate completeness
    completeness = 1.0 - null_ratio
    
    # Calculate weighted score
    weighted_sum = (
        weights.get('uniqueness', 0.4) * uniqueness_ratio +
        weights.get('completeness', 0.3) * completeness +
        weights.get('entropy', 0.2) * entropy_score +
        weights.get('pattern_bonus', 0.1) * pattern_bonus +
        weights.get('sequence_penalty', -0.1) * sequence_penalty
    )
    
    # Apply sigmoid normalization
    normalized_score = sigmoid(weighted_sum, steepness=6.0, midpoint=0.5)
    
    # Make decision
    decision = make_decision(normalized_score)
    
    # Create explanation
    explanation_parts = []
    if weights.get('uniqueness', 0) > 0:
        explanation_parts.append(f"uniqueness={uniqueness_ratio:.3f}*{weights.get('uniqueness', 0):.3f}")
    if weights.get('completeness', 0) > 0:
        explanation_parts.append(f"completeness={completeness:.3f}*{weights.get('completeness', 0):.3f}")
    if weights.get('entropy', 0) > 0:
        explanation_parts.append(f"entropy={entropy_score:.3f}*{weights.get('entropy', 0):.3f}")
    
    explanation = f"score=sigmoid({'+'.join(explanation_parts[:3])})={normalized_score:.3f}"
    
    # Feature importance for detailed explanation
    feature_importance = {
        'uniqueness': weights.get('uniqueness', 0) * uniqueness_ratio,
        'completeness': weights.get('completeness', 0) * completeness,
        'entropy': weights.get('entropy', 0) * entropy_score,
        'pattern_bonus': weights.get('pattern_bonus', 0) * pattern_bonus
    }
    
    detailed_explanation = explain_decision(normalized_score, decision, feature_importance)
    
    return {
        'score': normalized_score,
        'decision': decision,
        'explanation': explanation,
        'detailed_explanation': detailed_explanation,
        'features': {
            'uniqueness_ratio': uniqueness_ratio,
            'null_ratio': null_ratio,
            'completeness': completeness,
            'entropy_score': entropy_score,
            'pattern_bonus': pattern_bonus,
            'sequence_penalty': sequence_penalty
        }
    }


def calculate_fk_score_detailed(
    coverage_fk_to_pk: float,
    coverage_pk_to_fk: float,
    pk_uniqueness: float,
    fk_null_ratio: float,
    datatype_match: float,
    pattern_match: float,
    name_fuzzy_similarity: float,
    constraint_violation_rate: float,
    name_embedding_similarity: Optional[float] = None,
    value_embedding_similarity: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate weighted foreign key relationship score.
    
    Args:
        coverage_fk_to_pk: Ratio of FK values found in PK
        coverage_pk_to_fk: Ratio of PK values found in FK
        pk_uniqueness: Uniqueness ratio of the PK column
        fk_null_ratio: Ratio of null values in FK
        datatype_match: Data type compatibility score
        pattern_match: Pattern similarity score
        name_fuzzy_similarity: Column name similarity score
        constraint_violation_rate: Rate of constraint violations
        name_embedding_similarity: Semantic name similarity (optional)
        value_embedding_similarity: Value similarity (optional)
        weights: Custom weights (uses config defaults if None)
        
    Returns:
        Tuple of (final_score, explanation)
    """
    logger.info("START: calculate_fk_score_detailed called")
    logger.info(f"weights parameter type: {type(weights)}")
    
    if weights is None:
        weights = Config.scoring_weights
        
    logger.info(f"weights after config: {type(weights)}")
    
    # Prepare features
    features = {
        'coverage_fk_to_pk': coverage_fk_to_pk,
        'coverage_pk_to_fk': coverage_pk_to_fk,
        'pk_uniqueness': pk_uniqueness,
        'fk_completeness': 1.0 - fk_null_ratio,
        'datatype_match': datatype_match,
        'pattern_match': pattern_match,
        'name_fuzzy_similarity': name_fuzzy_similarity,
        'name_embedding_similarity': name_embedding_similarity or 0.0,
        'value_embedding_similarity': value_embedding_similarity or 0.0,
        'constraint_compliance': 1.0 - constraint_violation_rate
    }
    
    # Calculate weighted sum
    weighted_sum = 0.0
    explanation_parts = []
    feature_importance = {}  # Track feature contributions for explanation
    
    # Debug: Check types
    if not isinstance(features, dict):
        logger.error(f"Features is not a dict: {type(features)} = {features}")
        return {
            'score': 0.0,
            'decision': 'rejected',
            'explanation': f'Error: features is {type(features)}, not dict',
            'detailed_explanation': f'Technical error: expected dict, got {type(features)}',
            'features': {}
        }
    
    if not isinstance(weights, dict):
        logger.error(f"Weights is not a dict: {type(weights)} = {weights}")
        weights = Config.scoring_weights  # Fallback
    
    for feature_name, feature_value in features.items():
        weight = weights.get(feature_name, 0.0)
        contribution = weight * feature_value
        weighted_sum += contribution
        
        # Track feature importance
        feature_importance[feature_name] = contribution
        
        if weight > 0.01:  # Only explain significant contributions
            explanation_parts.append(f"{feature_name}={feature_value:.3f}*{weight:.3f}")
    
    # Apply sigmoid normalization
    raw_score = weighted_sum
    normalized_score = sigmoid(raw_score, steepness=8.0, midpoint=0.5)
    
    # Create explanation
    explanation = f"score=sigmoid({'+'.join(explanation_parts[:5])}+...)={normalized_score:.3f}"
    if len(explanation_parts) > 5:
        explanation += f" [{len(explanation_parts)} features]"
    
    # Make decision
    decision = make_decision(normalized_score)
    detailed_explanation = explain_decision(normalized_score, decision, feature_importance)
    
    return {
        'score': normalized_score,
        'decision': decision,
        'explanation': explanation,
        'detailed_explanation': detailed_explanation,
        'features': features
    }


def calculate_pk_score(
    uniqueness_ratio: float,
    null_ratio: float,
    entropy: float,
    pattern_type: str,
    monotonic_ratio: Optional[float] = None,
    cardinality: Optional[int] = None,
    total_rows: Optional[int] = None
) -> Tuple[float, str]:
    """
    Calculate primary key candidacy score.
    
    Args:
        uniqueness_ratio: Ratio of unique values
        null_ratio: Ratio of null values
        entropy: Shannon entropy of values
        pattern_type: Detected pattern type
        monotonic_ratio: Monotonic sequence ratio (for int sequences)
        cardinality: Number of unique values
        total_rows: Total number of rows
        
    Returns:
        Tuple of (score, explanation)
    """
    # Base score from uniqueness and nulls
    base_score = uniqueness_ratio * (1.0 - null_ratio)
    
    # Entropy bonus (higher entropy = better PK)
    entropy_normalized = min(entropy / 5.0, 1.0)  # Normalize entropy to [0,1]
    entropy_bonus = entropy_normalized * 0.2
    
    # Pattern type bonuses
    pattern_bonuses = {
        'uuid': 0.3,
        'int_sequence': 0.2,
        'email': 0.1,
        'monotonic': 0.15,
        'unknown': 0.0
    }
    pattern_bonus = pattern_bonuses.get(pattern_type, 0.0)
    
    # Monotonic sequence bonus
    monotonic_bonus = 0.0
    if monotonic_ratio is not None and pattern_type == 'int_sequence':
        monotonic_bonus = monotonic_ratio * 0.1
    
    # Cardinality bonus (more unique values = better PK, up to a point)
    cardinality_bonus = 0.0
    if cardinality is not None and total_rows is not None and total_rows > 0:
        cardinality_ratio = min(cardinality / total_rows, 1.0)
        if cardinality_ratio > 0.8:  # High cardinality is good
            cardinality_bonus = 0.1
    
    # Combine scores
    total_score = base_score + entropy_bonus + pattern_bonus + monotonic_bonus + cardinality_bonus
    
    # Apply sigmoid to keep in [0,1]
    final_score = sigmoid(total_score, steepness=5.0, midpoint=0.7)
    
    # Create explanation
    explanation = (
        f"PK_score=sigmoid(uniqueness={uniqueness_ratio:.3f}, "
        f"nulls={null_ratio:.3f}, entropy={entropy:.2f}, "
        f"pattern={pattern_type}+{pattern_bonus:.2f}) = {final_score:.3f}"
    )
    
    return final_score, explanation


def make_decision(score: float, accept_threshold: Optional[float] = None, 
                 reject_threshold: Optional[float] = None) -> str:
    """
    Make accept/reject/ambiguous decision based on score and thresholds.
    
    Args:
        score: Calculated score
        accept_threshold: Threshold above which to accept (uses config default if None)
        reject_threshold: Threshold below which to reject (uses config default if None)
        
    Returns:
        Decision string: 'accepted', 'rejected', or 'ambiguous'
    """
    if accept_threshold is None:
        accept_threshold = Config.decision_accept_threshold
    
    if reject_threshold is None:
        reject_threshold = Config.decision_reject_threshold
    
    if score >= accept_threshold:
        return 'accepted'
    elif score <= reject_threshold:
        return 'rejected'
    else:
        return 'ambiguous'


def explain_decision(score: float, decision: str, feature_importance: Optional[Dict[str, float]] = None) -> str:
    """
    Generate human-readable explanation for a decision.
    
    Args:
        score: Final score
        decision: Decision made
        feature_importance: Feature contributions (optional)
        
    Returns:
        Explanation string
    """
    base_explanation = f"Score: {score:.3f} → {decision}"
    
    if decision == 'accepted':
        reason = f"High confidence (≥{Config.decision_accept_threshold})"
    elif decision == 'rejected':
        reason = f"Low confidence (≤{Config.decision_reject_threshold})"
    else:
        reason = f"Uncertain ({Config.decision_reject_threshold} < score < {Config.decision_accept_threshold})"
    
    explanation = f"{base_explanation}. {reason}."
    
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        feature_str = ", ".join([f"{k}={v:.3f}" for k, v in top_features])
        explanation += f" Top factors: {feature_str}"
    
    return explanation


def validate_score_inputs(**kwargs) -> Dict[str, Any]:
    """
    Validate and normalize inputs for scoring functions.
    
    Args:
        **kwargs: Input parameters to validate
        
    Returns:
        Validated and normalized parameters
    """
    validated = {}
    
    # Define expected ranges for parameters
    ranges = {
        'coverage_fk_to_pk': (0.0, 1.0),
        'coverage_pk_to_fk': (0.0, 1.0),
        'pk_uniqueness': (0.0, 1.0),
        'fk_null_ratio': (0.0, 1.0),
        'uniqueness_ratio': (0.0, 1.0),
        'null_ratio': (0.0, 1.0),
        'datatype_match': (0.0, 1.0),
        'pattern_match': (0.0, 1.0),
        'name_fuzzy_similarity': (0.0, 1.0),
        'constraint_violation_rate': (0.0, 1.0)
    }
    
    for param, value in kwargs.items():
        if value is None:
            validated[param] = 0.0
            continue
            
        if param in ranges:
            min_val, max_val = ranges[param]
            if not isinstance(value, (int, float)):
                logger.warning(f"Invalid type for {param}: {type(value)}, using 0.0")
                validated[param] = 0.0
            elif value < min_val or value > max_val:
                logger.warning(f"Value {value} for {param} outside range [{min_val}, {max_val}], clamping")
                validated[param] = max(min_val, min(max_val, float(value)))
            else:
                validated[param] = float(value)
        else:
            validated[param] = value
    
    return validated


# Convenience functions for common scoring scenarios
def score_foreign_key_relationship(fk_stats: Dict[str, Any], pk_stats: Dict[str, Any], 
                                  name_similarity: float) -> Dict[str, Any]:
    """
    Score a foreign key relationship given column statistics.
    
    Args:
        fk_stats: Foreign key column statistics
        pk_stats: Primary key column statistics  
        name_similarity: Name similarity score
        
    Returns:
        Complete scoring result with score, decision, and explanation
    """
    # Extract required metrics
    coverage_fk_to_pk = fk_stats.get('coverage_to_pk', 0.0)
    coverage_pk_to_fk = fk_stats.get('coverage_from_pk', 0.0)
    pk_uniqueness = pk_stats.get('uniqueness_ratio', 0.0)
    fk_null_ratio = fk_stats.get('null_ratio', 0.0)
    datatype_match = 1.0 if fk_stats.get('dtype') == pk_stats.get('dtype') else 0.0
    pattern_match = fk_stats.get('pattern_match_score', 0.0)
    constraint_violation_rate = fk_stats.get('constraint_violation_rate', 0.0)
    
    # Validate inputs
    validated = validate_score_inputs(
        coverage_fk_to_pk=coverage_fk_to_pk,
        coverage_pk_to_fk=coverage_pk_to_fk,
        pk_uniqueness=pk_uniqueness,
        fk_null_ratio=fk_null_ratio,
        datatype_match=datatype_match,
        pattern_match=pattern_match,
        name_fuzzy_similarity=name_similarity,
        constraint_violation_rate=constraint_violation_rate
    )
    
    # Calculate score
    score, explanation = calculate_fk_score(**validated)
    decision = make_decision(score)
    detailed_explanation = explain_decision(score, decision)
    
    return {
        'score': score,
        'decision': decision,
        'explanation': explanation,
        'detailed_explanation': detailed_explanation,
        'features': validated
    }


def score_primary_key_candidate(column_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a primary key candidate given column statistics.
    
    Args:
        column_stats: Column statistics dictionary
        
    Returns:
        Complete scoring result with score, decision, and explanation
    """
    # Extract required metrics
    uniqueness_ratio = column_stats.get('uniqueness_ratio', 0.0)
    null_ratio = column_stats.get('null_ratio', 0.0)
    entropy = column_stats.get('entropy', 0.0)
    pattern_type = column_stats.get('pattern_type', 'unknown')
    monotonic_ratio = column_stats.get('monotonic_ratio')
    cardinality = column_stats.get('unique_count')
    total_rows = column_stats.get('total_count')
    
    # Calculate score
    try:
        score_result = calculate_pk_score(
            uniqueness_ratio=uniqueness_ratio,
            null_ratio=null_ratio,
            entropy=entropy,
            pattern_type=pattern_type,
            monotonic_ratio=monotonic_ratio,
            cardinality=cardinality,
            total_rows=total_rows
        )
        
        # Handle both tuple and dict returns
        if isinstance(score_result, tuple):
            score, explanation = score_result
        elif isinstance(score_result, dict):
            score = score_result.get('score', 0.0)
            explanation = score_result.get('explanation', 'No explanation')
        else:
            score = float(score_result) if score_result else 0.0
            explanation = 'Basic score calculation'
    except Exception as e:
        # Fallback scoring if calculate_pk_score fails
        score = uniqueness_ratio * 0.8 + (1.0 - null_ratio) * 0.2
        explanation = f"Fallback scoring (error: {e})"
    
    decision = make_decision(score)
    detailed_explanation = explain_decision(score, decision)
    
    return {
        'score': score,
        'decision': decision,
        'explanation': explanation,
        'detailed_explanation': detailed_explanation,
        'features': {
            'uniqueness_ratio': uniqueness_ratio,
            'null_ratio': null_ratio,
            'entropy': entropy,
            'pattern_type': pattern_type,
            'monotonic_ratio': monotonic_ratio
        }
    }