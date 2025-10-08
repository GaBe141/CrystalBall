#!/usr/bin/env python3
"""
Test script to verify all necessary libraries are installed and working.
"""

import importlib
import sys
import traceback


def test_core_libraries() -> bool:
    """Test core data science libraries."""
    print("🔍 Testing core libraries...")
    try:
        pd = importlib.import_module('pandas')
        np = importlib.import_module('numpy')
        matplotlib = importlib.import_module('matplotlib')
        sns = importlib.import_module('seaborn')
        plotly = importlib.import_module('plotly')
        print(f"✅ Pandas {pd.__version__}")
        print(f"✅ NumPy {np.__version__}")
        print(f"✅ Matplotlib {matplotlib.__version__}")
        print(f"✅ Seaborn {sns.__version__}")
        print(f"✅ Plotly {plotly.__version__}")
        return True
    except Exception as e:
        print(f"❌ Core libraries error: {e}")
        return False

def test_forecasting_libraries() -> bool:
    """Test forecasting-specific libraries."""
    print("\n🔮 Testing forecasting libraries...")
    success = True
    
    # Test statsmodels
    try:
        sm_base = importlib.import_module('statsmodels')
        # Ensure API is importable as well
        importlib.import_module('statsmodels.api')
        print(f"✅ Statsmodels {sm_base.__version__}")
    except Exception as e:
        print(f"❌ Statsmodels error: {e}")
        success = False
    
    # Test Prophet
    try:
        importlib.import_module('prophet')
        print("✅ Prophet imported successfully")
    except Exception as e:
        print(f"❌ Prophet error: {e}")
        success = False
    
    # Test LightGBM
    try:
        lgb = importlib.import_module('lightgbm')
        print(f"✅ LightGBM {lgb.__version__}")
    except Exception as e:
        print(f"❌ LightGBM error: {e}")
        success = False
    
    # Test NeuralProphet
    try:
        importlib.import_module('neuralprophet')
        print("✅ NeuralProphet imported successfully")
    except Exception as e:
        print(f"❌ NeuralProphet error: {e}")
        success = False
    
    # Test TBATS
    try:
        importlib.import_module('tbats')
        print("✅ TBATS imported successfully")
    except Exception as e:
        print(f"❌ TBATS error: {e}")
        success = False
    
    # Test Darts
    try:
        importlib.import_module('darts')
        print("✅ Darts imported successfully")
    except Exception as e:
        print(f"❌ Darts error: {e}")
        success = False
    
    # Test scikit-learn
    try:
        sklearn = importlib.import_module('sklearn')
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"❌ Scikit-learn error: {e}")
        success = False
    
    return success

def test_additional_libraries() -> bool:
    """Test additional useful libraries."""
    print("\n🛠️ Testing additional libraries...")
    success = True
    
    try:
        importlib.import_module('pmdarima')
        print("✅ pmdarima imported successfully")
    except Exception as e:
        print(f"❌ pmdarima error: {e}")
        success = False
    
    try:
        importlib.import_module('streamlit')
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit error: {e}")
        success = False
    
    try:
        importlib.import_module('pytest')
        print("✅ Pytest imported successfully")
    except Exception as e:
        print(f"❌ Pytest error: {e}")
        success = False
    
    return success

def test_crystalball_imports() -> bool:
    """Test CrystalBall project imports."""
    print("\n🔮 Testing CrystalBall project imports...")
    success = True
    
    try:
        cpi_mod = importlib.import_module('src.visualization.cpi_plot_generator')
        if hasattr(cpi_mod, 'CPIPlotGenerator'):
            print("✅ CPIPlotGenerator imported successfully")
        else:
            raise ImportError("CPIPlotGenerator not found in module")
    except Exception as e:
        print(f"❌ CPIPlotGenerator error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        success = False
    
    try:
        importlib.import_module('src.core.utils')
        print("✅ Core utils imported successfully")
    except Exception as e:
        print(f"❌ Core utils error: {e}")
        success = False
    
    try:
        importlib.import_module('src.models.advanced_models')
        print("✅ Advanced models imported successfully")
    except Exception as e:
        print(f"❌ Advanced models error: {e}")
        success = False
    
    return success

def test_simple_forecast() -> bool:
    """Test a simple forecasting example."""
    print("\n📊 Testing simple CPI forecast...")
    try:
        cpi_mod = importlib.import_module('src.visualization.cpi_plot_generator')
        CPIPlotGenerator = cpi_mod.CPIPlotGenerator
        # Create generator with synthetic data
        generator = CPIPlotGenerator()
        cpi_data = generator.load_cpi_data()
        
        print(f"✅ Generated synthetic CPI data: {len(cpi_data)} observations")
        print(f"✅ Data range: {cpi_data.index[0]} to {cpi_data.index[-1]}")
        
        # Test a simple model
        result = generator.fit_model('naive', test_size=12)
        if not result.get('error'):
            print("✅ Naive model fitted successfully")
        else:
            print(f"❌ Naive model error: {result['error']}")
        
        return True
    except Exception as e:
        print(f"❌ Simple forecast test error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main() -> int:
    """Run all tests."""
    print("🚀 Testing CrystalBall Installation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run all test functions
    tests = [
        test_core_libraries,
        test_forecasting_libraries,
        test_additional_libraries,
        test_crystalball_imports,
        test_simple_forecast
    ]
    
    for test_func in tests:
        if not test_func():
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your installation is ready.")
        print("\n📋 Summary:")
        print("• Core data science libraries: ✅")
        print("• Forecasting libraries: ✅")
        print("• CrystalBall project modules: ✅")
        print("• Simple forecast test: ✅")
        print("\n🎯 You can now run CPI forecasting models!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())