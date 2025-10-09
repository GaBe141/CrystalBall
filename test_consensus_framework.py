"""
Test script for Enhanced Consensus Framework
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.consensus import ConsensusEngine, EnhancedConsensusRunner, ConsensusValidator
from src.consensus.metrics import compute_consensus_metrics
from src.llm_consensus import HeuristicProvider


def test_consensus_engine() -> bool:
    """Test basic consensus engine functionality."""
    print("🧪 Testing Consensus Engine...")
    
    # Create a simple test case
    rankings_data = {
        'model': ['arima', 'ets', 'theta', 'naive'],
        'rmse': [0.15, 0.12, 0.18, 0.25],
        'mae': [0.12, 0.10, 0.15, 0.20],
        'mape': [5.2, 4.8, 6.1, 8.3]
    }
    rankings_df = pd.DataFrame(rankings_data)
    
    # Initialize engine with only heuristic provider for testing
    engine = ConsensusEngine(providers=[HeuristicProvider()])
    
    try:
        # Test consensus computation
        consensus_scores = engine.compute_consensus("test_cpi", rankings_df)
        
        print(f"✅ Generated {len(consensus_scores)} consensus scores")
        
        if consensus_scores:
            top_model = consensus_scores[0]
            print(f"🏆 Top model: {top_model.model} (score: {top_model.weighted_score:.3f})")
            print(f"📊 Confidence: {top_model.confidence:.3f}, Variance: {top_model.variance:.3f}")
        
        # Test summary
        summary = engine.get_consensus_summary()
        print(f"📈 Engine summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consensus engine test failed: {e}")
        return False


def test_enhanced_runner() -> bool:
    """Test enhanced consensus runner integration."""
    print("\n🧪 Testing Enhanced Consensus Runner...")
    
    try:
        # Check if processed data exists
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            print("⚠️  No processed data directory found - creating test case")
            
            # Create test data
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            test_rankings = {
                'model': ['arima', 'ets', 'theta', 'naive', 'lstm'],
                'rmse': [0.15, 0.12, 0.18, 0.25, 0.14],
                'mae': [0.12, 0.10, 0.15, 0.20, 0.11],
                'cv_rmse': [0.16, 0.13, 0.19, 0.26, 0.15]
            }
            
            test_df = pd.DataFrame(test_rankings)
            test_file = processed_dir / "test_cpi_rankings.csv"
            test_df.to_csv(test_file, index=False)
            print(f"📁 Created test file: {test_file}")
        
        # Initialize runner with only heuristic for testing
        runner = EnhancedConsensusRunner(
            providers=[HeuristicProvider()],
            methods=["weighted_average", "median"]
        )
        
        # Run enhanced consensus
        results = runner.run_enhanced_consensus(
            rankings_dir=str(processed_dir),
            output_dir="exports/test_consensus"
        )
        
        print(f"✅ Enhanced consensus completed: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced runner test failed: {e}")
        return False


def test_validator() -> bool:
    """Test consensus validator."""
    print("\n🧪 Testing Consensus Validator...")
    
    try:
        # Create test data
        test_data = []
        
        for i in range(3):
            rankings_data = {
                'model': ['arima', 'ets', 'theta'],
                'rmse': [0.15 + i*0.01, 0.12 + i*0.01, 0.18 + i*0.01],
                'mae': [0.12 + i*0.01, 0.10 + i*0.01, 0.15 + i*0.01]
            }
            
            test_data.append({
                'series_name': f'test_series_{i}',
                'rankings_df': pd.DataFrame(rankings_data),
                'actual_best': 'ets'  # ETS has lowest errors
            })
        
        # Initialize validator
        engine = ConsensusEngine(providers=[HeuristicProvider()])
        validator = ConsensusValidator(engine)
        
        # Run validation
        validation_result = validator.cross_validate_consensus(test_data)
        
        print("✅ Validation completed")
        print(f"📊 Accuracy: {validation_result.accuracy:.3f}")
        print(f"🎯 Method performance: {validation_result.method_performance}")
        
        # Generate report
        report = validator.generate_validation_report()
        print(f"📄 Generated validation report (length: {len(report)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        return False


def test_metrics() -> bool:
    """Test metrics computation."""
    print("\n🧪 Testing Consensus Metrics...")
    
    try:
        # Create mock consensus history
        consensus_history = [
            {
                'series_name': 'test_1',
                'timestamp': '2024-01-01T00:00:00',
                'consensus_scores': [
                    {'model': 'arima', 'score': 0.8, 'confidence': 0.9, 'variance': 0.1},
                    {'model': 'ets', 'score': 0.7, 'confidence': 0.8, 'variance': 0.15}
                ]
            },
            {
                'series_name': 'test_2',
                'timestamp': '2024-01-02T00:00:00',
                'consensus_scores': [
                    {'model': 'ets', 'score': 0.85, 'confidence': 0.85, 'variance': 0.12},
                    {'model': 'arima', 'score': 0.75, 'confidence': 0.75, 'variance': 0.18}
                ]
            }
        ]
        
        # Compute metrics
        metrics = compute_consensus_metrics(consensus_history)
        
        print("✅ Computed metrics")
        print(f"🎯 Confidence score: {metrics.confidence_score:.3f}")
        print(f"🌈 Diversity score: {metrics.diversity_score:.3f}")
        print(f"📊 Overall score: {metrics.overall_score():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def main() -> bool:
    """Run all tests."""
    print("🚀 Starting Enhanced Consensus Framework Tests")
    print("=" * 60)
    
    tests = [
        test_consensus_engine,
        test_enhanced_runner,
        test_validator,
        test_metrics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    
    for i, (test, result) in enumerate(zip(tests, results, strict=True)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    total_passed = sum(results)
    print(f"\n🎯 Overall: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("🎉 All tests passed! Framework is ready for integration.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return total_passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)