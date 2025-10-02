"""
Configuration module for Adel-Lite detailed analysis.
Provides configurable thresholds, weights, and sampling strategies.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdelLiteConfig:
    """
    Configuration class for Adel-Lite enhanced detection.
    Contains all configurable parameters with validation.
    """
    
    # === VERSION ===
    # Performance optimization settings
    sample_size: int = 10000
    enable_bloom_filters: bool = True
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None  # Auto-detect
    chunk_size: int = 1000
    bloom_filter_fp_rate: float = 0.01
    
    # Advanced fuzzy matching settings
    fuzzy_matching_method: str = 'ratio'  # 'ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'
    fuzzy_matching_threshold: float = 0.6
    
    version: str = "0.2.0-detailed"
    
    # === PRIMARY KEY DETECTION CONFIGURATION ===
    pk_uniqueness_threshold: float = 0.999  # Allow up to 0.1% duplicates
    pk_null_tolerance: float = 0.02  # Allow up to 2% nulls
    pk_entropy_threshold: float = 2.0  # Minimum entropy for PK candidates
    pk_min_cardinality: int = 2  # Minimum unique values for PK
    pk_max_null_count: int = None  # Absolute max nulls (None = use percentage)
    
    # === FOREIGN KEY DETECTION ===
    fk_coverage_threshold: float = 0.8  # Minimum FK→PK coverage
    fk_reverse_coverage_threshold: float = 0.1  # Minimum PK→FK coverage
    fk_null_tolerance: float = 0.3  # Allow up to 30% nulls in FK
    fk_constraint_violation_threshold: float = 0.2  # Max 20% violations
    
    # === NAME SIMILARITY ===
    name_similarity_threshold: float = 0.7
    name_fuzzy_algorithm: str = 'ratio'  # rapidfuzz algorithm
    
    # === SCORING WEIGHTS ===
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'coverage_fk_to_pk': 0.25,          # FK values found in PK
        'coverage_pk_to_fk': 0.15,          # PK values found in FK  
        'pk_uniqueness': 0.20,              # How unique is the PK
        'fk_completeness': 0.10,             # 1 - FK null ratio
        'datatype_match': 0.05,             # Data types match
        'pattern_match': 0.05,              # Pattern similarity
        'name_fuzzy_similarity': 0.10,      # String similarity
        'name_embedding_similarity': 0.05,  # Semantic similarity (future)
        'value_embedding_similarity': 0.03, # Value similarity (future)
        'constraint_compliance': 0.02       # 1 - violation rate
    })
    
    # === DECISION THRESHOLDS ===
    decision_accept_threshold: float = 0.85   # Above this = accept
    decision_reject_threshold: float = 0.25   # Below this = reject
    # Between = ambiguous
    
    # === SAMPLING CONFIGURATION ===
    sample_strategy: str = 'random'  # 'random', 'stratified', 'head', 'tail'
    sample_size: int = 10000  # Max rows to sample for analysis
    profile_sample_size: int = 10000  # Max rows for profiling analysis
    sample_seed: int = 42  # Random seed for reproducibility
    
    # === PERFORMANCE OPTIMIZATION ===
    use_bloom_filters: bool = True  # Use Bloom filters for large FK lookups
    bloom_filter_capacity: int = 1000000  # Expected number of elements
    bloom_filter_error_rate: float = 0.01  # 1% false positive rate
    parallel_processing: bool = False  # Enable multiprocessing (future)
    max_workers: int = 4  # Number of worker processes
    
    # === PATTERN DETECTION ===
    pattern_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'uuid': {
            'regex': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            'min_samples': 3,
            'confidence_threshold': 0.8
        },
        'email': {
            'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'min_samples': 2,
            'confidence_threshold': 0.7
        },
        'phone': {
            'regex': r'^[\+]?[1-9][\d]{0,15}$',
            'min_samples': 2,
            'confidence_threshold': 0.6
        },
        'url': {
            'regex': r'^https?://[^\s/$.?#].[^\s]*$',
            'min_samples': 2,
            'confidence_threshold': 0.7
        },
        'int_sequence': {
            'monotonic_threshold': 0.9,  # 90% monotonic
            'min_samples': 5,
            'gap_tolerance': 0.1  # Allow 10% gaps
        }
    })
    
    # === ENTROPY CALCULATION ===
    entropy_sample_size: int = 1000  # Max samples for entropy calculation
    entropy_bins: int = 50  # Number of bins for numeric entropy
    
    # === EXPORT CONFIGURATION ===
    export_include_features: bool = True  # Include full feature dict
    export_include_explanations: bool = True  # Include decision explanations
    export_precision: int = 4  # Decimal places for scores
    
    # === LOGGING ===
    log_level: str = 'INFO'
    log_performance: bool = False  # Log timing information
    log_decisions: bool = True  # Log PK/FK decisions
    
    # === VALIDATION ===
    strict_mode: bool = False  # Strict validation vs relaxed
    validate_config: bool = True  # Validate config on load
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.validate_config:
            self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        # Validate thresholds are in [0, 1]
        thresholds = [
            ('pk_uniqueness_threshold', self.pk_uniqueness_threshold),
            ('pk_null_tolerance', self.pk_null_tolerance),
            ('fk_coverage_threshold', self.fk_coverage_threshold),
            ('fk_reverse_coverage_threshold', self.fk_reverse_coverage_threshold),
            ('fk_null_tolerance', self.fk_null_tolerance),
            ('fk_constraint_violation_threshold', self.fk_constraint_violation_threshold),
            ('name_similarity_threshold', self.name_similarity_threshold),
            ('decision_accept_threshold', self.decision_accept_threshold),
            ('decision_reject_threshold', self.decision_reject_threshold)
        ]
        
        for name, value in thresholds:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        
        # Validate decision thresholds order
        if self.decision_reject_threshold >= self.decision_accept_threshold:
            raise ValueError("decision_reject_threshold must be < decision_accept_threshold")
        
        # Validate scoring weights sum (should be close to 1.0)
        weight_sum = sum(self.scoring_weights.values())
        if abs(weight_sum - 1.0) > 0.1:
            logger.warning(f"Scoring weights sum to {weight_sum:.3f}, should be close to 1.0")
        
        # Validate sample size
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        
        logger.info("Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'pk_settings': {
                'uniqueness_threshold': self.pk_uniqueness_threshold,
                'null_tolerance': self.pk_null_tolerance,
                'entropy_threshold': self.pk_entropy_threshold,
                'min_cardinality': self.pk_min_cardinality
            },
            'fk_settings': {
                'coverage_threshold': self.fk_coverage_threshold,
                'reverse_coverage_threshold': self.fk_reverse_coverage_threshold,
                'null_tolerance': self.fk_null_tolerance,
                'constraint_violation_threshold': self.fk_constraint_violation_threshold
            },
            'scoring': {
                'weights': self.scoring_weights,
                'decision_thresholds': {
                    'accept': self.decision_accept_threshold,
                    'reject': self.decision_reject_threshold
                }
            },
            'sampling': {
                'strategy': self.sample_strategy,
                'size': self.sample_size,
                'seed': self.sample_seed
            },
            'performance': {
                'use_bloom_filters': self.use_bloom_filters,
                'parallel_processing': self.parallel_processing,
                'max_workers': self.max_workers
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdelLiteConfig':
        """Create config from dictionary."""
        # Flatten nested structure for constructor
        kwargs = {}
        
        if 'pk_settings' in config_dict:
            pk = config_dict['pk_settings']
            kwargs.update({
                'pk_uniqueness_threshold': pk.get('uniqueness_threshold', 0.999),
                'pk_null_tolerance': pk.get('null_tolerance', 0.02),
                'pk_entropy_threshold': pk.get('entropy_threshold', 2.0),
                'pk_min_cardinality': pk.get('min_cardinality', 2)
            })
        
        if 'fk_settings' in config_dict:
            fk = config_dict['fk_settings']
            kwargs.update({
                'fk_coverage_threshold': fk.get('coverage_threshold', 0.8),
                'fk_reverse_coverage_threshold': fk.get('reverse_coverage_threshold', 0.1),
                'fk_null_tolerance': fk.get('null_tolerance', 0.3),
                'fk_constraint_violation_threshold': fk.get('constraint_violation_threshold', 0.2)
            })
        
        if 'scoring' in config_dict:
            scoring = config_dict['scoring']
            if 'weights' in scoring:
                kwargs['scoring_weights'] = scoring['weights']
            if 'decision_thresholds' in scoring:
                dt = scoring['decision_thresholds']
                kwargs.update({
                    'decision_accept_threshold': dt.get('accept', 0.85),
                    'decision_reject_threshold': dt.get('reject', 0.25)
                })
        
        if 'sampling' in config_dict:
            sampling = config_dict['sampling']
            kwargs.update({
                'sample_strategy': sampling.get('strategy', 'random'),
                'sample_size': sampling.get('size', 10000),
                'sample_seed': sampling.get('seed', 42)
            })
        
        return cls(**kwargs)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        if self.validate_config:
            self._validate()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        self.update(**config_dict)
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        default_config = AdelLiteConfig(validate_config=False)
        for key in default_config.__dict__:
            if not key.startswith('_'):
                setattr(self, key, getattr(default_config, key))
        
        if self.validate_config:
            self._validate()


# Global configuration instance
Config = AdelLiteConfig()


def load_config_from_file(file_path: str) -> None:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        file_path: Path to configuration file
    """
    global Config
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    import json
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        elif file_path.endswith(('.yml', '.yaml')):
            import yaml
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml")
        
        Config = AdelLiteConfig.from_dict(config_dict)
        logger.info(f"Configuration loaded from {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {e}")
        raise


def save_config_to_file(file_path: str) -> None:
    """
    Save current configuration to JSON file.
    
    Args:
        file_path: Path to save configuration
    """
    import json
    
    try:
        with open(file_path, 'w') as f:
            json.dump(Config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {file_path}: {e}")
        raise


def get_config() -> AdelLiteConfig:
    """Get the global configuration instance."""
    return Config


def set_config(**kwargs) -> None:
    """Update global configuration parameters."""
    Config.update(**kwargs)


# Convenience functions for common configurations
def set_strict_mode():
    """Enable strict mode with conservative thresholds."""
    Config.update(
        pk_uniqueness_threshold=1.0,
        pk_null_tolerance=0.0,
        fk_coverage_threshold=0.95,
        fk_null_tolerance=0.1,
        decision_accept_threshold=0.9,
        decision_reject_threshold=0.1,
        strict_mode=True
    )


def set_relaxed_mode():
    """Enable relaxed mode with permissive thresholds."""
    Config.update(
        pk_uniqueness_threshold=0.95,
        pk_null_tolerance=0.1,
        fk_coverage_threshold=0.7,
        fk_null_tolerance=0.5,
        decision_accept_threshold=0.7,
        decision_reject_threshold=0.3,
        strict_mode=False
    )


def set_performance_mode():
    """Enable performance optimizations."""
    Config.update(
        use_bloom_filters=True,
        sample_size=5000,
        parallel_processing=True,
        log_performance=True
    )


# Export for easier importing
__all__ = ['Config', 'AdelLiteConfig', 'get_config', 'set_config',
           'set_strict_mode', 'set_relaxed_mode', 'set_performance_mode']
