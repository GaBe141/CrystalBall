#!/usr/bin/env python3
"""
CPI Forecasting CLI Tool

Command-line interface for the CPI forecasting plot generator.
"""

import argparse
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.cpi_plot_generator import CPIPlotGenerator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description='CPI Forecasting Plot Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cpi_forecast_cli.py --model arima
  python cpi_forecast_cli.py --compare naive drift arima
  python cpi_forecast_cli.py --rank naive drift ses arima ets
  python cpi_forecast_cli.py --dashboard
  python cpi_forecast_cli.py --interactive
        """
    )
    
    # Data options
    parser.add_argument('--data', type=str, help='Path to CPI data file')
    parser.add_argument('--output', type=str, default='data/processed/visualizations',
                       help='Output directory for plots (default: data/processed/visualizations)')
    
    # Model options
    parser.add_argument('--model', type=str, 
                       help='Generate plot for single model (e.g., arima, ets, prophet)')
    parser.add_argument('--compare', nargs='+', metavar='MODEL',
                       help='Compare multiple models')
    parser.add_argument('--rank', nargs='+', metavar='MODEL',
                       help='Rank multiple models by performance')
    
    # Forecasting options
    parser.add_argument('--test-size', type=int, default=12,
                       help='Number of periods to hold out for testing (default: 12)')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of cross-validation folds for ranking (default: 3)')
    
    # Output options
    parser.add_argument('--dashboard', action='store_true',
                       help='Generate interactive HTML dashboard')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CPIPlotGenerator(data_path=args.data, output_dir=args.output)
    
    # List models
    if args.list_models:
        generator.print_available_models()
        return
    
    # Interactive mode
    if args.interactive:
        generator.run_interactive_mode()
        return
    
    # Load data
    print("📊 Loading CPI data...")
    cpi_data = generator.load_cpi_data()
    print(f"✅ Loaded {len(cpi_data)} CPI observations")
    
    try:
        # Single model
        if args.model:
            print(f"🔄 Generating {args.model.upper()} forecast...")
            fig = generator.generate_single_model_plot(
                model_name=args.model, 
                test_size=args.test_size
            )
            plt.close(fig)
            print(f"✅ {args.model.upper()} plot saved to {args.output}")
        
        # Model comparison
        elif args.compare:
            print(f"🔄 Comparing models: {', '.join([m.upper() for m in args.compare])}")
            fig = generator.generate_comparison_plot(
                model_names=args.compare,
                test_size=args.test_size
            )
            plt.close(fig)
            print(f"✅ Comparison plot saved to {args.output}")
        
        # Model ranking
        elif args.rank:
            print(f"🔄 Ranking models: {', '.join([m.upper() for m in args.rank])}")
            print("This may take a few minutes for cross-validation...")
            fig = generator.generate_model_ranking_plot(
                model_names=args.rank,
                test_size=args.test_size,
                cv_folds=args.cv_folds
            )
            plt.close(fig)
            print(f"✅ Ranking plot saved to {args.output}")
        
        # Dashboard
        elif args.dashboard:
            print("🔄 Generating interactive dashboard...")
            html_path = generator.generate_interactive_dashboard()
            print(f"✅ Dashboard saved: {html_path}")
            print("💡 Open the HTML file in your browser")
        
        else:
            # Default: show help
            parser.print_help()
            print("\n💡 Use --interactive for interactive mode or specify a specific operation.")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()