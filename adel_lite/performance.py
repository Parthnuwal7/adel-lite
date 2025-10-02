"""
Performance optimization utilities for Adel-Lite.
Includes Bloom filters, sampling strategies, and parallel processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import mmh3  # MurmurHash3 for Bloom filters
from rapidfuzz import fuzz, process
import random
from .config import Config

logger = logging.getLogger(__name__)


class BloomFilter:
    """
    Memory-efficient Bloom filter for fast FK lookup preprocessing.
    """
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter.
        
        Args:
            expected_items: Expected number of items to insert
            false_positive_rate: Desired false positive rate (0.01 = 1%)
        """
        self.expected_items = expected_items
        self.fp_rate = false_positive_rate
        
        # Calculate optimal size and hash functions
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)
        
        # Initialize bit array
        self.bit_array = np.zeros(self.size, dtype=bool)
        self.item_count = 0
        
        logger.info(f"Bloom filter initialized: {self.size} bits, {self.hash_count} hash functions")
    
    def _optimal_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        return int(-n * np.log(p) / (np.log(2) ** 2))
    
    def _optimal_hash_count(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        return max(1, int((m / n) * np.log(2)))
    
    def _hash(self, item: Any, seed: int) -> int:
        """Generate hash for item with given seed."""
        if isinstance(item, str):
            return mmh3.hash(item, seed) % self.size
        else:
            # Convert to string first
            return mmh3.hash(str(item), seed) % self.size
    
    def add(self, item: Any) -> None:
        """Add item to Bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
        self.item_count += 1
    
    def __contains__(self, item: Any) -> bool:
        """Check if item might be in the set (no false negatives)."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True
    
    def add_batch(self, items: List[Any]) -> None:
        """Add multiple items efficiently."""
        for item in items:
            self.add(item)
    
    def estimated_false_positive_rate(self) -> float:
        """Calculate current false positive rate."""
        if self.item_count == 0:
            return 0.0
        
        # Probability that a bit is still 0
        prob_zero = (1 - 1/self.size) ** (self.hash_count * self.item_count)
        # False positive rate
        return (1 - prob_zero) ** self.hash_count


class SamplingStrategy:
    """
    Intelligent sampling strategies for large datasets.
    """
    
    @staticmethod
    def random_sample(df: pd.DataFrame, sample_size: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Random sampling with optional seed."""
        if len(df) <= sample_size:
            return df
        
        if seed:
            return df.sample(n=sample_size, random_state=seed)
        return df.sample(n=sample_size)
    
    @staticmethod
    def stratified_sample(df: pd.DataFrame, column: str, sample_size: int, 
                         seed: Optional[int] = None) -> pd.DataFrame:
        """Stratified sampling based on a column."""
        if len(df) <= sample_size:
            return df
        
        # Get value counts
        value_counts = df[column].value_counts()
        n_unique = len(value_counts)
        
        if n_unique == 0:
            return SamplingStrategy.random_sample(df, sample_size, seed)
        
        # Calculate samples per stratum
        samples_per_stratum = max(1, sample_size // n_unique)
        remaining_samples = sample_size - (samples_per_stratum * n_unique)
        
        sampled_dfs = []
        
        # Sample from each stratum
        for value in value_counts.index:
            stratum = df[df[column] == value]
            stratum_size = min(len(stratum), samples_per_stratum)
            
            if remaining_samples > 0:
                stratum_size += 1
                remaining_samples -= 1
            
            if seed:
                stratum_sample = stratum.sample(n=stratum_size, random_state=seed)
            else:
                stratum_sample = stratum.sample(n=stratum_size)
            
            sampled_dfs.append(stratum_sample)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    @staticmethod
    def systematic_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Systematic sampling with fixed interval."""
        if len(df) <= sample_size:
            return df
        
        interval = len(df) // sample_size
        start = random.randint(0, interval - 1)
        indices = range(start, len(df), interval)[:sample_size]
        
        return df.iloc[indices]
    
    @staticmethod
    def adaptive_sample(df: pd.DataFrame, target_size: int = None) -> pd.DataFrame:
        """
        Adaptive sampling based on dataset characteristics.
        """
        if target_size is None:
            target_size = Config.sample_size
        
        if len(df) <= target_size:
            return df
        
        # For very large datasets, use systematic sampling
        if len(df) > 1_000_000:
            logger.info(f"Large dataset ({len(df)} rows), using systematic sampling")
            return SamplingStrategy.systematic_sample(df, target_size)
        
        # For medium datasets, use random sampling
        elif len(df) > 100_000:
            logger.info(f"Medium dataset ({len(df)} rows), using random sampling")
            return SamplingStrategy.random_sample(df, target_size)
        
        # For smaller datasets, no sampling needed
        else:
            return df


class FuzzyMatcher:
    """
    High-performance fuzzy string matching using rapidfuzz.
    """
    
    @staticmethod
    def calculate_similarity(str1: str, str2: str, method: str = 'ratio') -> float:
        """
        Calculate string similarity using rapidfuzz.
        
        Args:
            str1, str2: Strings to compare
            method: Similarity method ('ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio')
        
        Returns:
            Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0.0
        
        try:
            if method == 'ratio':
                score = fuzz.ratio(str1, str2)
            elif method == 'partial_ratio':
                score = fuzz.partial_ratio(str1, str2)
            elif method == 'token_sort_ratio':
                score = fuzz.token_sort_ratio(str1, str2)
            elif method == 'token_set_ratio':
                score = fuzz.token_set_ratio(str1, str2)
            else:
                score = fuzz.ratio(str1, str2)
            
            return score / 100.0  # Convert to 0-1 range
        
        except Exception as e:
            logger.warning(f"Error in fuzzy matching: {e}")
            return 0.0
    
    @staticmethod
    def find_best_matches(query: str, choices: List[str], limit: int = 5, 
                         threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find best matching strings from a list.
        
        Args:
            query: String to match
            choices: List of candidate strings
            limit: Maximum number of matches to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (match, score) tuples
        """
        if not query or not choices:
            return []
        
        try:
            # Use rapidfuzz process for efficient matching
            matches = process.extract(query, choices, limit=limit, scorer=fuzz.ratio)
            
            # Filter by threshold and convert to 0-1 range
            filtered_matches = [
                (match, score / 100.0) 
                for match, score, _ in matches 
                if score / 100.0 >= threshold
            ]
            
            return filtered_matches
        
        except Exception as e:
            logger.warning(f"Error in batch fuzzy matching: {e}")
            return []


class ParallelProcessor:
    """
    Parallel processing utilities for computationally intensive operations.
    """
    
    @staticmethod
    def parallel_map(func, items: List[Any], max_workers: Optional[int] = None, 
                    use_processes: bool = False) -> List[Any]:
        """
        Apply function to items in parallel.
        
        Args:
            func: Function to apply
            items: List of items to process
            max_workers: Maximum number of workers (None = auto)
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        
        Returns:
            List of results
        """
        if not items:
            return []
        
        if max_workers is None:
            max_workers = min(len(items), mp.cpu_count())
        
        # For small datasets, use sequential processing
        if len(items) < 100:
            return [func(item) for item in items]
        
        try:
            if use_processes:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(func, items))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(func, items))
            
            return results
        
        except Exception as e:
            logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
            return [func(item) for item in items]
    
    @staticmethod
    def parallel_apply_chunks(df: pd.DataFrame, func, chunk_size: int = 10000, 
                             max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Apply function to DataFrame chunks in parallel.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            max_workers: Maximum number of workers
        
        Returns:
            Concatenated result DataFrame
        """
        if len(df) <= chunk_size:
            return func(df)
        
        # Split into chunks
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        results = ParallelProcessor.parallel_map(func, chunks, max_workers, use_processes=True)
        
        # Concatenate results
        return pd.concat(results, ignore_index=True)


class PerformanceOptimizer:
    """
    Main class for applying performance optimizations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.bloom_filters: Dict[str, BloomFilter] = {}
    
    def optimize_fk_detection(self, fk_df: pd.DataFrame, pk_df: pd.DataFrame, 
                             pk_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Optimize FK detection using Bloom filters and sampling.
        
        Args:
            fk_df: Foreign key DataFrame
            pk_df: Primary key DataFrame  
            pk_columns: List of PK column names
        
        Returns:
            Tuple of (optimized_fk_df, optimized_pk_df, optimization_metadata)
        """
        metadata = {
            'original_fk_rows': len(fk_df),
            'original_pk_rows': len(pk_df),
            'optimizations_applied': []
        }
        
        # Apply sampling if datasets are large
        optimized_fk_df = fk_df
        optimized_pk_df = pk_df
        
        if len(fk_df) > Config.sample_size:
            optimized_fk_df = SamplingStrategy.adaptive_sample(fk_df, Config.sample_size)
            metadata['optimizations_applied'].append('fk_sampling')
            metadata['fk_sample_size'] = len(optimized_fk_df)
        
        if len(pk_df) > Config.sample_size:
            optimized_pk_df = SamplingStrategy.adaptive_sample(pk_df, Config.sample_size)
            metadata['optimizations_applied'].append('pk_sampling')
            metadata['pk_sample_size'] = len(optimized_pk_df)
        
        # Create Bloom filters for PK columns
        for pk_col in pk_columns:
            if pk_col in optimized_pk_df.columns:
                pk_values = optimized_pk_df[pk_col].dropna().unique()
                if len(pk_values) > 1000:  # Only use Bloom filter for larger sets
                    bloom_filter = BloomFilter(len(pk_values))
                    bloom_filter.add_batch(pk_values.tolist())
                    self.bloom_filters[f"{pk_col}"] = bloom_filter
                    metadata['optimizations_applied'].append(f'bloom_filter_{pk_col}')
        
        metadata['final_fk_rows'] = len(optimized_fk_df)
        metadata['final_pk_rows'] = len(optimized_pk_df)
        
        logger.info(f"FK detection optimized: {metadata}")
        return optimized_fk_df, optimized_pk_df, metadata
    
    def check_bloom_filter(self, column_name: str, value: Any) -> bool:
        """
        Check if value might exist in PK column using Bloom filter.
        
        Returns:
            True if value might exist (no false negatives)
            False if value definitely doesn't exist
        """
        bloom_key = f"{column_name}"
        if bloom_key in self.bloom_filters:
            return value in self.bloom_filters[bloom_key]
        return True  # No filter available, assume might exist
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about applied optimizations."""
        stats = {
            'bloom_filters_active': len(self.bloom_filters),
            'bloom_filter_details': {}
        }
        
        for key, bloom_filter in self.bloom_filters.items():
            stats['bloom_filter_details'][key] = {
                'size': bloom_filter.size,
                'items': bloom_filter.item_count,
                'estimated_fp_rate': bloom_filter.estimated_false_positive_rate()
            }
        
        return stats


def benchmark_performance(func, *args, iterations: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Benchmark function performance.
    
    Args:
        func: Function to benchmark
        *args, **kwargs: Function arguments
        iterations: Number of iterations to run
    
    Returns:
        Performance statistics
    """
    import time
    
    times = []
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        times.append(end_time - start_time)
        results.append(result)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'iterations': iterations,
        'results_consistent': len(set(str(r) for r in results)) == 1 if results else True
    }


# Performance-optimized versions of common operations
def fast_intersection(list1: List[Any], list2: List[Any]) -> Set[Any]:
    """Fast intersection using sets."""
    return set(list1) & set(list2)


def fast_jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Fast Jaccard similarity calculation."""
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def memory_efficient_groupby(df: pd.DataFrame, column: str, 
                           agg_func: str = 'count') -> pd.DataFrame:
    """Memory-efficient groupby operation."""
    try:
        # Use categorical for string columns to save memory
        if df[column].dtype == 'object':
            df[column] = df[column].astype('category')
        
        return df.groupby(column, observed=True).size().reset_index(name=agg_func)
    
    except Exception as e:
        logger.warning(f"Memory-efficient groupby failed: {e}")
        return df.groupby(column).size().reset_index(name=agg_func)