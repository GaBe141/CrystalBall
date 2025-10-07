#!/usr/bin/env python3
"""
Demo script for CPI Plot Generator

This script demonstrates how to use the CPI forecasting plot generator
with different models and visualization options.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from visualization.cpi_plot_generator import CPIPlotGenerator, quick_cpi_forecast

def main() -> None:
    """Run the CPI plot generator demo."""
    print("🔮 CPI Forecasting Plot Generator Demo")
    print("=" * 50)
    
    # Initialize the plot generator
    generator = CPIPlotGenerator(output_dir="data/processed/visualizations")
    
    # Load CPI data (will create synthetic data if no real data found)
    print("📊 Loading CPI data...")
    cpi_data = generator.load_cpi_data()
    print(f"✅ Loaded {len(cpi_data)} CPI observations")
    print(f"📅 Period: {cpi_data.index[0].strftime('%Y-%m')} to {cpi_data.index[-1].strftime('%Y-%m')}")
    
    # Demo 1: Single model plots
    print("\n🔄 Demo 1: Single Model Forecasts")
    print("-" * 40)
    
    simple_models = ['naive', 'drift', 'arima']
    
    for model in simple_models:
        try:
            print(f"Generating {model.upper()} forecast...")
            fig = generator.generate_single_model_plot(
                model_name=model, 
                test_size=12, 
                save_plot=True
            )
            plt.close(fig)  # Close to save memory
            print(f"✅ {model.upper()} plot generated")
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
    
    # Demo 2: Model comparison
    print("\n🔄 Demo 2: Model Comparison")
    print("-" * 40)
    
    try:
        comparison_models = ['naive', 'drift', 'ses']
        print(f"Comparing models: {', '.join([m.upper() for m in comparison_models])}")
        fig = generator.generate_comparison_plot(
            model_names=comparison_models,
            test_size=12,
            save_plot=True
        )
        plt.close(fig)
        print("✅ Comparison plot generated")
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
    
    # Demo 3: Model ranking
    print("\n🔄 Demo 3: Model Performance Ranking")
    print("-" * 40)
    
    try:
        ranking_models = ['naive', 'drift', 'ses']
        print("Performing cross-validation ranking...")
        fig = generator.generate_model_ranking_plot(
            model_names=ranking_models,
            test_size=6,
            cv_folds=3,
            save_plot=True
        )
        plt.close(fig)
        print("✅ Ranking plot generated")
    except Exception as e:
        print(f"❌ Error in ranking: {e}")
    
    # Demo 4: Interactive dashboard
    print("\n🔄 Demo 4: Interactive Dashboard")
    print("-" * 40)
    
    try:
        html_path = generator.generate_interactive_dashboard()
        print(f"✅ Interactive dashboard generated: {html_path}")
        print("💡 Open the HTML file in your browser for interactive features")
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
    
    # Demo 5: Quick forecast function
    print("\n🔄 Demo 5: Quick Forecast")
    print("-" * 40)
    
    try:
        print("Generating quick ARIMA forecast...")
        fig = quick_cpi_forecast(model_name='arima', show_plot=False)
        plt.close(fig)
        print("✅ Quick forecast generated")
    except Exception as e:
        print(f"❌ Error in quick forecast: {e}")
    
    print("\n🎉 Demo completed!")
    print(f"📁 Check the output directory: {generator.output_dir}")
    print("🖼️  All generated plots and dashboard are saved there.")

def interactive_demo() -> None:
    """Run interactive demo where user can choose options."""
    generator = CPIPlotGenerator()
    generator.run_interactive_mode()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        main()