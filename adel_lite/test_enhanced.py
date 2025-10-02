"""
Comprehensive test suite for enhanced Adel-Lite detection functions.
Includes edge cases, configuration validation, and performance benchmarks.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import os
from typing import Dict, List, Any
import unittest.mock as mock

# Import modules to test
from adel_lite.config import Config, AdelLiteConfig
from adel_lite.scoring_detailed import (
    calculate_fk_score_detailed, calculate_pk_score_detailed, 
    sigmoid, make_decision, explain_decision
)
from adel_lite.pk_detection_detailed import (
    detect_primary_keys_detailed, analyze_column_quality, 
    detect_pattern_type, calculate_entropy
)
from adel_lite.fk_detection_detailed import (
    detect_foreign_keys_detailed, calculate_bidirectional_coverage,
    analyze_constraint_violations, analyze_datatype_compatibility
)
from adel_lite.profile_detailed import profile_detailed
from adel_lite.map_relationships_detailed import analyze_table_relationships
from adel_lite.export_detailed import export_schema_graph_detailed, SchemaExporter
from adel_lite.performance import (
    BloomFilter, SamplingStrategy, FuzzyMatcher, PerformanceOptimizer,
    benchmark_performance
)


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        assert Config.pk_uniqueness_threshold == 0.999
        assert Config.fk_coverage_threshold == 0.8
        assert Config.name_similarity_threshold == 0.7
        assert isinstance(Config.scoring_weights, dict)
        assert len(Config.scoring_weights) >= 10
    
    def test_config_dataclass(self):
        """Test AdelLiteConfig dataclass."""
        config = AdelLiteConfig()
        assert config.pk_uniqueness_threshold == 0.999
        assert config.validate()
        
        # Test invalid values
        config.pk_uniqueness_threshold = 1.5  # Invalid
        assert not config.validate()
    
    def test_config_file_io(self):
        """Test configuration file save/load."""
        config = AdelLiteConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_file(f.name)
            
            # Load and verify
            loaded_config = AdelLiteConfig.load_from_file(f.name)
            assert loaded_config.pk_uniqueness_threshold == config.pk_uniqueness_threshold
            
            os.unlink(f.name)
    
    def test_config_update_from_dict(self):
        """Test configuration updates from dictionary."""
        original_threshold = Config.pk_uniqueness_threshold
        
        Config.update_from_dict({
            'pk_uniqueness_threshold': 0.95,
            'sample_size': 5000
        })
        
        assert Config.pk_uniqueness_threshold == 0.95
        assert Config.sample_size == 5000
        
        # Reset
        Config.pk_uniqueness_threshold = original_threshold


class TestScoringSystem:
    """Test scoring and decision making."""
    
    def test_sigmoid_function(self):
        """Test sigmoid normalization."""
        # Test boundary conditions
        assert sigmoid(0.0) == 0.5
        assert sigmoid(-10.0) < 0.1
        assert sigmoid(10.0) > 0.9
        
        # Test monotonicity
        assert sigmoid(0.3) < sigmoid(0.7)
    
    def test_make_decision(self):
        """Test decision thresholds."""
        assert make_decision(0.9) == 'accepted'
        assert make_decision(0.5) == 'ambiguous'
        assert make_decision(0.1) == 'rejected'
    
    def test_fk_score_calculation(self):
        """Test FK score calculation."""
        result = calculate_fk_score_detailed(
            coverage_fk_to_pk=0.8,
            coverage_pk_to_fk=0.9,
            pk_uniqueness=1.0,
            fk_null_ratio=0.0,
            datatype_match=1.0,
            pattern_match=1.0,
            name_fuzzy_similarity=0.8,
            constraint_violation_rate=0.0
        )
        
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'decision' in result
        assert 'explanation' in result
        assert 0.0 <= result['score'] <= 1.0
        assert result['decision'] in ['accepted', 'rejected', 'ambiguous']
    
    def test_pk_score_calculation(self):
        """Test PK score calculation."""
        result = calculate_pk_score_detailed(
            uniqueness_ratio=1.0,
            null_ratio=0.0,
            entropy_score=0.9,
            pattern_bonus=0.1,
            sequence_penalty=0.0
        )
        
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'decision' in result
        assert 0.0 <= result['score'] <= 1.0


class TestPrimaryKeyDetection:
    """Test primary key detection functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.users_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'username': ['alice', 'bob', 'charlie', 'diana', 'eve'],
            'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                     'diana@example.com', 'eve@example.com'],
            'age': [25, 30, 35, 28, 32]
        })
        
        self.bad_pk_df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3],  # Not unique
            'name': ['a', 'b', 'c', 'd', 'e'],
            'value': [None, None, None, 1, 2]  # Many nulls
        })
    
    def test_detect_primary_keys_basic(self):
        """Test basic PK detection."""
        results = detect_primary_keys_detailed(self.users_df, 'users')
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that we found good candidates
        scores = [r['score'] for r in results]
        assert max(scores) > 0.8  # At least one good candidate
    
    def test_detect_primary_keys_poor_candidates(self):
        """Test PK detection with poor candidates."""
        results = detect_primary_keys_detailed(self.bad_pk_df, 'bad_table')
        
        # Should still return results but with lower scores
        assert isinstance(results, list)
        if results:
            # No candidate should have a very high score
            scores = [r['score'] for r in results]
            assert max(scores) < 0.9
    
    def test_pattern_detection(self):
        """Test pattern detection in columns."""
        # Test UUID pattern
        assert detect_pattern_type(['550e8400-e29b-41d4-a716-446655440000']) == 'uuid'
        
        # Test email pattern
        assert detect_pattern_type(['user@example.com', 'test@test.org']) == 'email'
        
        # Test phone pattern
        assert detect_pattern_type(['+1-555-123-4567', '555-987-6543']) == 'phone'
        
        # Test unknown pattern
        assert detect_pattern_type(['random', 'text', 'values']) == 'unknown'
    
    def test_entropy_calculation(self):
        """Test entropy calculation."""
        # High entropy (uniform distribution)
        high_entropy_values = list(range(100))
        high_entropy = calculate_entropy(high_entropy_values)
        
        # Low entropy (repeated values)
        low_entropy_values = [1] * 95 + [2, 3, 4, 5, 6]
        low_entropy = calculate_entropy(low_entropy_values)
        
        assert high_entropy > low_entropy
        assert high_entropy > 0.8
        assert low_entropy < 0.5


class TestForeignKeyDetection:
    """Test foreign key detection functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.users_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'username': ['alice', 'bob', 'charlie', 'diana', 'eve']
        })
        
        self.orders_df = pd.DataFrame({
            'order_id': [101, 102, 103, 104],
            'user_id': [1, 2, 1, 3],  # Foreign key to users
            'amount': [100.0, 200.0, 150.0, 300.0]
        })
        
        self.bad_fk_df = pd.DataFrame({
            'id': [1, 2, 3],
            'user_id': [99, 98, 97],  # No matches with users
            'data': ['a', 'b', 'c']
        })
    
    def test_detect_foreign_keys_valid(self):
        """Test FK detection with valid relationships."""
        results = detect_foreign_keys_detailed(
            self.orders_df, 'orders',
            self.users_df, 'users',
            ['user_id']
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Should find the user_id relationship
        user_id_rels = [r for r in results if r['fk_column'] == 'user_id']
        assert len(user_id_rels) > 0
        
        # Check relationship quality
        best_rel = max(user_id_rels, key=lambda x: x['score'])
        assert best_rel['score'] > 0.5
    
    def test_detect_foreign_keys_invalid(self):
        """Test FK detection with invalid relationships."""
        results = detect_foreign_keys_detailed(
            self.bad_fk_df, 'bad_fk',
            self.users_df, 'users',
            ['user_id']
        )
        
        # Should find relationships but with low scores
        if results:
            scores = [r['score'] for r in results]
            assert max(scores) < 0.5
    
    def test_bidirectional_coverage(self):
        """Test bidirectional coverage calculation."""
        fk_values = [1, 2, 1, 3, 4]
        pk_values = [1, 2, 3, 5, 6]
        
        coverage = calculate_bidirectional_coverage(fk_values, pk_values)
        
        assert 'fk_to_pk_coverage' in coverage
        assert 'pk_to_fk_coverage' in coverage
        assert 'intersection_size' in coverage
        
        # Check values are reasonable
        assert 0.0 <= coverage['fk_to_pk_coverage'] <= 1.0
        assert 0.0 <= coverage['pk_to_fk_coverage'] <= 1.0
    
    def test_constraint_violations(self):
        """Test constraint violation analysis."""
        fk_values = [1, 2, 3, 99]  # 99 is a violation
        pk_values = [1, 2, 3]
        
        violations = analyze_constraint_violations(fk_values, pk_values)
        
        assert 'violation_count' in violations
        assert 'violation_ratio' in violations
        assert violations['violation_count'] == 1
        assert violations['violation_ratio'] == 0.25


class TestProfileDetailed:
    """Test detailed profiling functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 
                     'diana@test.com', 'eve@test.com'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000.0, 60000.0, None, 55000.0, 65000.0]
        })
    
    def test_profile_detailed_basic(self):
        """Test basic profiling functionality."""
        profile = profile_detailed(self.test_df, 'test_table')
        
        assert isinstance(profile, dict)
        assert 'table_name' in profile
        assert 'primary_key_candidates' in profile
        assert 'column_profiles' in profile
        assert 'table_statistics' in profile
        
        # Check column profiles
        assert len(profile['column_profiles']) == len(self.test_df.columns)
        
        # Check each column has required fields
        for col_name, col_profile in profile['column_profiles'].items():
            assert 'data_type' in col_profile
            assert 'null_ratio' in col_profile
            assert 'uniqueness_ratio' in col_profile
    
    def test_profile_with_relationships(self):
        """Test profiling with relationship analysis."""
        profile = profile_detailed(self.test_df, 'test_table', include_relationships=True)
        
        # Should include relationship analysis (empty in this case)
        assert 'relationship_analysis' in profile


class TestExportDetailed:
    """Test export functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.tables = {
            'users': pd.DataFrame({
                'user_id': [1, 2, 3],
                'username': ['alice', 'bob', 'charlie'],
                'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com']
            }),
            'orders': pd.DataFrame({
                'order_id': [101, 102, 103],
                'user_id': [1, 2, 1],
                'amount': [100.0, 200.0, 150.0]
            })
        }
    
    def test_export_schema_graph_detailed(self):
        """Test schema graph export."""
        schema_graph = export_schema_graph_detailed(self.tables)
        
        assert isinstance(schema_graph, dict)
        assert 'metadata' in schema_graph
        assert 'tables' in schema_graph
        assert 'relationships' in schema_graph
        assert 'summary' in schema_graph
        assert 'quality_metrics' in schema_graph
        
        # Check metadata
        metadata = schema_graph['metadata']
        assert 'created_at' in metadata
        assert 'adel_lite_version' in metadata
        assert metadata['table_count'] == 2
        
        # Check tables
        assert len(schema_graph['tables']) == 2
        assert 'users' in schema_graph['tables']
        assert 'orders' in schema_graph['tables']
    
    def test_schema_exporter_file_output(self):
        """Test file output functionality."""
        exporter = SchemaExporter()
        schema_graph = exporter.export_schema_graph_detailed(self.tables)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            success = exporter.save_to_file(schema_graph, f.name, 'json')
            assert success
            
            # Verify file was created and contains valid JSON
            with open(f.name, 'r') as read_f:
                loaded_data = json.load(read_f)
                assert loaded_data == schema_graph
            
            os.unlink(f.name)
    
    def test_summary_report_generation(self):
        """Test summary report generation."""
        exporter = SchemaExporter()
        schema_graph = exporter.export_schema_graph_detailed(self.tables)
        
        report = exporter.export_summary_report(schema_graph)
        
        assert isinstance(report, str)
        assert 'Database Schema Analysis Report' in report
        assert 'Summary Statistics' in report
        assert 'Quality Metrics' in report


class TestPerformanceOptimizations:
    """Test performance optimization components."""
    
    def test_bloom_filter(self):
        """Test Bloom filter functionality."""
        bloom = BloomFilter(expected_items=1000)
        
        # Add items
        test_items = ['item1', 'item2', 'item3', 123, 456]
        for item in test_items:
            bloom.add(item)
        
        # Test membership (no false negatives)
        for item in test_items:
            assert item in bloom
        
        # Test batch operations
        batch_items = ['batch1', 'batch2', 'batch3']
        bloom.add_batch(batch_items)
        
        for item in batch_items:
            assert item in bloom
    
    def test_sampling_strategies(self):
        """Test various sampling strategies."""
        df = pd.DataFrame({
            'id': range(1000),
            'category': ['A'] * 300 + ['B'] * 300 + ['C'] * 400,
            'value': np.random.randn(1000)
        })
        
        # Random sampling
        sample = SamplingStrategy.random_sample(df, 100)
        assert len(sample) == 100
        
        # Stratified sampling
        stratified = SamplingStrategy.stratified_sample(df, 'category', 100)
        assert len(stratified) <= 100
        
        # Systematic sampling
        systematic = SamplingStrategy.systematic_sample(df, 100)
        assert len(systematic) == 100
        
        # Adaptive sampling
        adaptive = SamplingStrategy.adaptive_sample(df, 100)
        assert len(adaptive) == 100
    
    def test_fuzzy_matcher(self):
        """Test fuzzy string matching."""
        # Test similarity calculation
        similarity = FuzzyMatcher.calculate_similarity('test', 'test')
        assert similarity == 1.0
        
        similarity = FuzzyMatcher.calculate_similarity('test', 'testing')
        assert 0.5 < similarity < 1.0
        
        # Test batch matching
        choices = ['apple', 'application', 'apply', 'orange', 'grape']
        matches = FuzzyMatcher.find_best_matches('app', choices, limit=3)
        
        assert len(matches) <= 3
        assert all(isinstance(match, tuple) and len(match) == 2 for match in matches)
    
    def test_performance_optimizer(self):
        """Test performance optimizer."""
        optimizer = PerformanceOptimizer()
        
        fk_df = pd.DataFrame({'id': range(10000), 'ref_id': range(10000)})
        pk_df = pd.DataFrame({'id': range(5000), 'data': ['test'] * 5000})
        
        optimized_fk, optimized_pk, metadata = optimizer.optimize_fk_detection(
            fk_df, pk_df, ['id']
        )
        
        assert isinstance(optimized_fk, pd.DataFrame)
        assert isinstance(optimized_pk, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert 'optimizations_applied' in metadata


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframes(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully without crashing
        results = detect_primary_keys_detailed(empty_df, 'empty_table')
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames."""
        single_row = pd.DataFrame({'id': [1], 'name': ['test']})
        
        results = detect_primary_keys_detailed(single_row, 'single_row')
        assert isinstance(results, list)
        # Single row should have unique values, but low confidence due to sample size
    
    def test_all_null_columns(self):
        """Test handling of columns with all null values."""
        null_df = pd.DataFrame({
            'id': [1, 2, 3],
            'null_col': [None, None, None],
            'mixed_col': [1, None, 3]
        })
        
        results = detect_primary_keys_detailed(null_df, 'null_test')
        
        # Should not recommend all-null column as PK
        null_col_results = [r for r in results if r['column'] == 'null_col']
        if null_col_results:
            assert null_col_results[0]['score'] < 0.5
    
    def test_large_cardinality_columns(self):
        """Test handling of very high cardinality columns."""
        large_card_df = pd.DataFrame({
            'id': range(10000),
            'uuid': [f'uuid-{i}' for i in range(10000)],
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='1min')
        })
        
        # Should handle without performance issues
        results = detect_primary_keys_detailed(large_card_df, 'large_card')
        assert isinstance(results, list)


class TestBenchmarks:
    """Performance benchmarks for optimization validation."""
    
    def test_pk_detection_benchmark(self):
        """Benchmark PK detection performance."""
        # Create test dataset
        test_df = pd.DataFrame({
            'id': range(10000),
            'name': [f'name_{i}' for i in range(10000)],
            'email': [f'user{i}@example.com' for i in range(10000)],
            'value': np.random.randn(10000)
        })
        
        # Benchmark the function
        benchmark_result = benchmark_performance(
            detect_primary_keys_detailed,
            test_df, 'benchmark_table',
            iterations=3
        )
        
        assert 'mean_time' in benchmark_result
        assert benchmark_result['mean_time'] > 0
        assert benchmark_result['results_consistent']
        
        print(f"PK Detection Benchmark: {benchmark_result['mean_time']:.3f}s average")
    
    def test_fk_detection_benchmark(self):
        """Benchmark FK detection performance."""
        fk_df = pd.DataFrame({
            'id': range(5000),
            'ref_id': np.random.randint(0, 1000, 5000)
        })
        
        pk_df = pd.DataFrame({
            'id': range(1000),
            'data': [f'data_{i}' for i in range(1000)]
        })
        
        benchmark_result = benchmark_performance(
            detect_foreign_keys_detailed,
            fk_df, 'fk_table', pk_df, 'pk_table', ['id'],
            iterations=3
        )
        
        assert 'mean_time' in benchmark_result
        print(f"FK Detection Benchmark: {benchmark_result['mean_time']:.3f}s average")


def run_all_tests():
    """Run all tests and report results."""
    import time
    
    print("=" * 60)
    print("ADEL-LITE ENHANCED TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestConfiguration,
        TestScoringSystem,
        TestPrimaryKeyDetection,
        TestForeignKeyDetection,
        TestProfileDetailed,
        TestExportDetailed,
        TestPerformanceOptimizations,
        TestEdgeCases,
        TestBenchmarks
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    start_time = time.time()
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        test_instance = test_class()
        
        # Get test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Run setup if it exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {e}")
    
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Execution Time: {end_time - start_time:.2f}s")
    
    if failed_tests:
        print("\nFAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)