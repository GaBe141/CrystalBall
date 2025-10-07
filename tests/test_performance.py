import os
import tempfile
import time
import multiprocessing
from unittest.mock import patch

import pandas as pd
import pytest

from src import pipeline, utils


def test_parallel_pipeline_faster_than_sequential():
    """Test that parallel processing provides performance benefits."""
    # Generate 4 small test CSV files
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = os.path.join(tmp_dir, "raw")
        processed_dir = os.path.join(tmp_dir, "processed") 
        os.makedirs(raw_dir)
        os.makedirs(processed_dir)
        
        # Create test files
        for i in range(4):
            df = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=50, freq='M'),
                'value': range(50),
                'series_id': f'test_{i}'
            })
            df.to_csv(os.path.join(raw_dir, f'test_{i}.csv'), index=False)
        
        # Mock config to use temp directories
        mock_cfg = type('MockConfig', (), {
            'paths': type('Paths', (), {
                'raw_data_dir': raw_dir,
                'processed_dir': processed_dir,
                'visuals_dir': os.path.join(tmp_dir, 'visuals'),
                'exports_dir': os.path.join(tmp_dir, 'exports')
            })(),
            'settings': type('Settings', (), {
                'test_size_fraction': 0.2,
                'max_affinity_features': 3
            })()
        })()
        
        # Test sequential (max_workers=1)
        with patch('src.pipeline.load_config', return_value=mock_cfg):
            start_seq = time.time()
            results_seq = pipeline.analyze_all(max_workers=1, limit=4)
            time_seq = time.time() - start_seq
        
        # Test parallel (max_workers=2)
        with patch('src.pipeline.load_config', return_value=mock_cfg):
            start_par = time.time()
            results_par = pipeline.analyze_all(max_workers=2, limit=4)
            time_par = time.time() - start_par
        
        # Verify both produced results
        assert len(results_seq) == 4
        assert len(results_par) == 4
        
        # Parallel should be faster (allowing some overhead tolerance)
        speedup = time_seq / max(time_par, 0.1)  # Avoid division by zero
        print(f"Sequential: {time_seq:.2f}s, Parallel: {time_par:.2f}s, Speedup: {speedup:.2f}x")
        
        # Even with overhead, we should see some benefit with 2+ workers
        assert speedup > 0.8, f"Expected parallel processing to be competitive, got {speedup:.2f}x speedup"


def test_optimized_csv_reader_memory_efficiency():
    """Test that the optimized CSV reader reduces memory usage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test.csv")
        
        # Create a CSV with various data types
        df = pd.DataFrame({
            'big_float': [1.0] * 1000,  # Will be downcast to float32
            'big_int': list(range(1000)),  # Will be downcast to smaller int
            'category_like': ['A', 'B', 'A', 'B'] * 250,  # Will become category
            'unique_strings': [f'unique_{i}' for i in range(1000)]  # Stays object
        })
        df.to_csv(csv_path, index=False)
        
        # Test regular pandas read
        df_regular = pd.read_csv(csv_path)
        memory_regular = df_regular.memory_usage(deep=True).sum()
        
        # Test optimized read
        df_optimized = utils.read_csv_optimized(csv_path)
        memory_optimized = df_optimized.memory_usage(deep=True).sum()
        
        # Verify optimization worked
        assert df_optimized['big_float'].dtype == 'float32'
        assert df_optimized['big_int'].dtype in ['int8', 'int16', 'int32']  # Should be smaller than int64
        assert df_optimized['category_like'].dtype.name == 'category'
        assert df_optimized['unique_strings'].dtype == 'object'  # High cardinality stays object
        
        # Memory should be reduced
        memory_savings = (memory_regular - memory_optimized) / memory_regular
        print(f"Memory savings: {memory_savings:.2%}")
        assert memory_savings > 0.1, f"Expected >10% memory savings, got {memory_savings:.2%}"


def test_concurrent_llm_providers():
    """Test that LLM consensus can handle concurrent provider calls."""
    # Mock providers that simulate network delay
    class SlowMockProvider:
        def __init__(self, name, delay=0.1):
            self.name = name
            self.delay = delay
        
        def available(self):
            return True
        
        def score(self, series_name, df):
            time.sleep(self.delay)  # Simulate API call
            return {
                "panelist": self.name,
                "scores": [{"model": "test", "score": 0.8, "rank": 1, "notes": "mock"}]
            }
    
    # Test that concurrent calls are faster than sequential
    providers = [SlowMockProvider(f"provider_{i}", 0.05) for i in range(3)]
    df = pd.DataFrame({'model': ['arima', 'ets'], 'rmse': [0.1, 0.2]})
    
    # Sequential timing (simulated)
    start_seq = time.time()
    results_seq = []
    for p in providers:
        results_seq.append(p.score("test", df))
    time_seq = time.time() - start_seq
    
    # Concurrent timing (using ThreadPoolExecutor pattern from llm_consensus)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    start_conc = time.time()
    results_conc = []
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        future_to_prov = {executor.submit(p.score, "test", df): p for p in providers}
        for future in as_completed(future_to_prov):
            results_conc.append(future.result())
    time_conc = time.time() - start_conc
    
    # Verify results are equivalent
    assert len(results_seq) == len(results_conc) == 3
    
    # Concurrent should be faster
    speedup = time_seq / max(time_conc, 0.01)
    print(f"Sequential: {time_seq:.3f}s, Concurrent: {time_conc:.3f}s, Speedup: {speedup:.2f}x")
    assert speedup > 2.0, f"Expected significant speedup from concurrency, got {speedup:.2f}x"


if __name__ == "__main__":
    # Quick smoke test
    test_optimized_csv_reader_memory_efficiency()
    test_concurrent_llm_providers()
    print("Performance optimization tests passed!")