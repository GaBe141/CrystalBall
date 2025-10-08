#!/usr/bin/env python3
"""
Quick CPI Forecasting Demo
Test all the installed forecasting libraries.
"""

from typing import Any

from src.visualization.cpi_plot_generator import CPIPlotGenerator


def _test_models(
    generator: CPIPlotGenerator, models_to_test: list[str]
) -> dict[str, dict[str, Any]]:
    """Run model fits and print basic metrics for a small set of models."""
    results: dict[str, dict[str, Any]] = {}
    print(f"\n🧪 Testing models: {', '.join(models_to_test)}")
    for model_name in models_to_test:
        print(f"\n📈 Testing {model_name.upper()} model...")
        try:
            result = generator.fit_model(model_name, test_size=12)
            if result.get('error'):
                print(f"   ❌ Error: {result['error']}")
            else:
                print("   ✅ Success!")
                if result.get('mae') is not None:
                    print(f"      MAE: {result['mae']:.4f}")
                if result.get('rmse') is not None:
                    print(f"      RMSE: {result['rmse']:.4f}")
                if result.get('mape') is not None:
                    print(f"      MAPE: {result['mape']:.2f}%")
                results[model_name] = result
        except Exception as e:  # noqa: BLE001 - broad for demo UX
            print(f"   ❌ Exception: {e}")
    return results

def main() -> None:
    """Run a quick CPI forecasting demonstration."""
    print("🔮 CPI Forecasting Demo")
    print("=" * 50)
    
    # Initialize the plot generator
    generator = CPIPlotGenerator()
    
    # Load data (will create synthetic data if no real data available)
    print("📊 Loading CPI data...")
    cpi_data = generator.load_cpi_data()
    start_str = cpi_data.index[0].strftime('%Y-%m')
    end_str = cpi_data.index[-1].strftime('%Y-%m')
    print(
        f"✅ Loaded {len(cpi_data)} observations from {start_str} to {end_str}"
    )
    
    # Show available models
    print("\n🤖 Available models:")
    generator.print_available_models()
    
    # Test a few key models
    models_to_test = ['naive', 'drift', 'arima', 'ets']
    results = _test_models(generator, models_to_test)
    
    # Generate a comparison plot
    if len(results) > 1:
        print(f"\n📊 Generating comparison plot for {len(results)} models...")
        try:
            generator.generate_comparison_plot(
                list(results.keys()), 
                test_size=12, 
                save_plot=True
            )
            print(f"   ✅ Plot saved to: {generator.output_dir}")
        except Exception as e:
            print(f"   ❌ Plot generation error: {e}")
    
    # Generate individual model plot
    if results:
        best_model = list(results.keys())[0]
        print(f"\n📈 Generating individual plot for {best_model.upper()} model...")
        try:
            generator.generate_single_model_plot(
                best_model,
                test_size=12,
                save_plot=True
            )
            print(f"   ✅ Plot saved to: {generator.output_dir}")
        except Exception as e:
            print(f"   ❌ Individual plot error: {e}")
    
    print("\n🎉 Demo complete!")
    print(f"Check the '{generator.output_dir}' directory for generated plots.")
    print("\n💡 To add Kwartz model:")
    print("   1. Implement fit_kwartz_model() function")
    print("   2. Add 'kwartz' to available_models in CPIPlotGenerator")
    print("   3. Update the model dispatch logic")
    print(
        "   4. Run comparison with:"
        " generator.generate_comparison_plot(['kwartz', 'arima', 'ets'])"
    )

if __name__ == "__main__":
    main()