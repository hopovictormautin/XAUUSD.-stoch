"""
Script to run parameter optimization for the XAUUSD Trading Bot.

This script provides a command-line interface for optimizing the
parameters of the XAUUSD cent account scalping bot.
"""

import argparse
import time
import os
import json
from xauusd_bot import XAUUSDCentScalpingBot
from utils.reporting import generate_optimization_report

def main():
    """Main function to run parameter optimization"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run XAUUSD Trading Bot Parameter Optimization')
    
    parser.add_argument('--data', type=str, default='xau.csv',
                       help='Path to price data CSV file')
    parser.add_argument('--timeframe', type=str, default='5T',
                       help='Timeframe for analysis (e.g. 1T, 5T, 15T, 1H)')
    parser.add_argument('--param-ranges', type=str, default=None,
                       help='Path to JSON file with parameter ranges')
    parser.add_argument('--output-dir', type=str, default='results/optimization',
                       help='Directory to save optimization results')
    parser.add_argument('--metric', type=str, default='profit_factor',
                       help='Metric to optimize (profit_factor, sharpe_ratio, etc.)')
    parser.add_argument('--combinations', type=int, default=100,
                       help='Maximum number of parameter combinations to test')
    parser.add_argument('--walk-forward', action='store_true',
                       help='Use walk-forward validation')
    parser.add_argument('--train-test-split', type=float, default=0.7,
                       help='Train/test split ratio for walk-forward validation')
    parser.add_argument('--initial-capital', type=float, default=1000.0,
                       help='Initial capital in USD')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define default parameter ranges
    default_param_ranges = {
        "fast_ma_period": [5, 8, 10, 12, 15],
        "slow_ma_period": [50, 100, 150, 200],
        "stoch_k_period": [5, 7, 9, 14],
        "stoch_upper_level": [70, 75, 80, 85],
        "stoch_lower_level": [15, 20, 25, 30],
        "take_profit_points": [50, 75, 100],
        "stop_loss_points": [30, 45, 60],
        "use_adx_filter": [True, False],
        "trailing_stop": [True, False]
    }
    
    # Load parameter ranges from file if provided
    if args.param_ranges:
        try:
            with open(args.param_ranges, 'r') as f:
                param_ranges = json.load(f)
            print(f"Loaded parameter ranges from {args.param_ranges}")
        except Exception as e:
            print(f"Error loading parameter ranges file: {e}")
            print("Using default parameter ranges instead.")
            param_ranges = default_param_ranges
    else:
        print("Using default parameter ranges.")
        param_ranges = default_param_ranges
    
    # Create bot instance
    bot = XAUUSDCentScalpingBot(initial_capital=int(args.initial_capital * 100))
    
    # Run optimization
    print(f"Running optimization on {args.timeframe} timeframe using data from: {args.data}")
    print(f"Optimizing for metric: {args.metric}")
    print(f"Testing up to {args.combinations} parameter combinations")
    if args.walk_forward:
        print(f"Using walk-forward validation with {args.train_test_split:.1f}/{1-args.train_test_split:.1f} split")
    
    start_time = time.time()
    
    results = bot.optimize_parameters(
        args.data,
        param_ranges,
        use_timeframe=args.timeframe,
        num_combinations=args.combinations,
        optimization_metric=args.metric,
        use_walk_forward=args.walk_forward,
        train_test_split=args.train_test_split
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Optimization completed in {duration:.2f} seconds")
    
    # Generate timestamp for output files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save optimization results
    if results:
        # Save all results to JSON
        results_file = os.path.join(args.output_dir, f"optimization_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved all optimization results to {results_file}")
        
        # Save best parameters to JSON
        best_params_file = os.path.join(args.output_dir, f"best_params_{timestamp}.json")
        with open(best_params_file, 'w') as f:
            json.dump(results[0]["parameters"], f, indent=2)
        print(f"Saved best parameters to {best_params_file}")
        
        # Generate detailed report
        report_file = os.path.join(args.output_dir, f"optimization_report_{timestamp}.txt")
        generate_optimization_report(results, output_file=report_file)
        
        # Print best parameters summary
        print("\n=== Best Parameters ===")
        for param, value in results[0]["parameters"].items():
            print(f"{param}: {value}")
    else:
        print("No optimization results generated. Check for errors.")

if __name__ == "__main__":
    main()