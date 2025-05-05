"""
Main script for XAUUSD Trading Bot.

This script provides a unified command-line interface for running all operations
with the XAUUSD cent account scalping bot.
"""

import argparse
import os
import json
import time
from xauusd_bot import XAUUSDCentScalpingBot

def load_parameters(param_file):
    """Load parameters from JSON file"""
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded parameters from {param_file}")
        return params
    except Exception as e:
        print(f"Error loading parameters file: {e}")
        return {}

def run_backtest(args):
    """Run backtest with specified parameters"""
    print("\n=== Running Backtest ===")
    
    # Load parameters if specified
    params = {}
    if args.params:
        params = load_parameters(args.params)
    
    # Set initial capital
    params['initial_capital'] = int(args.initial_capital * 100)  # Convert to cents
    
    # Create bot instance
    bot = XAUUSDCentScalpingBot(**params)
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run backtest
    print(f"Running backtest on {args.timeframe} timeframe using data from: {args.data}")
    start_time = time.time()
    
    performance = bot.backtest(args.data, use_timeframe=args.timeframe)
    
    end_time = time.time()
    print(f"Backtest completed in {end_time - start_time:.2f} seconds")
    
    # Generate plots if requested
    if not args.no_plots:
        bot.plot_results()
    
    # Generate report
    if performance and args.output:
        bot.generate_trade_report(args.output)
        print(f"Report saved to {args.output}")
    
    # Print summary
    print_performance_summary(performance)
    
    return performance

def run_optimization(args):
    """Run parameter optimization"""
    print("\n=== Running Parameter Optimization ===")
    
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
    
    # Load parameter ranges if specified
    param_ranges = default_param_ranges
    if args.param_ranges:
        try:
            with open(args.param_ranges, 'r') as f:
                param_ranges = json.load(f)
            print(f"Loaded parameter ranges from {args.param_ranges}")
        except Exception as e:
            print(f"Error loading parameter ranges file: {e}")
            print("Using default parameter ranges instead.")
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create bot instance
    bot = XAUUSDCentScalpingBot(initial_capital=int(args.initial_capital * 100))
    
    # Run optimization
    print(f"Running optimization on {args.timeframe} timeframe")
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
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Generate timestamp for output files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save results if requested
    if results and args.output_dir:
        # Save all results
        results_file = os.path.join(args.output_dir, f"optimization_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved all optimization results to {results_file}")
        
        # Save best parameters
        best_params_file = os.path.join(args.output_dir, f"best_params_{timestamp}.json")
        with open(best_params_file, 'w') as f:
            json.dump(results[0]["parameters"], f, indent=2)
        print(f"Saved best parameters to {best_params_file}")
        
        # Generate report
        from utils.reporting import generate_optimization_report
        report_file = os.path.join(args.output_dir, f"optimization_report_{timestamp}.txt")
        generate_optimization_report(results, output_file=report_file)
    
    return results

def validate_optimized_parameters(args):
    """Run validation backtest with optimized parameters"""
    print("\n=== Running Validation Backtest ===")
    
    if not args.params:
        print("Error: Parameter file required for validation")
        return None
    
    # Load optimized parameters
    params = load_parameters(args.params)
    if not params:
        return None
    
    # Add initial capital
    params['initial_capital'] = int(args.initial_capital * 100)
    
    # Create bot instance
    bot = XAUUSDCentScalpingBot(**params)
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run backtest
    print(f"Running validation backtest on {args.timeframe} timeframe")
    
    performance = bot.backtest(args.data, use_timeframe=args.timeframe)
    
    # Generate plots if requested
    if not args.no_plots:
        bot.plot_results()
    
    # Generate report
    if performance and args.output:
        bot.generate_trade_report(args.output)
        print(f"Validation report saved to {args.output}")
    
    # Print summary
    print_performance_summary(performance)
    
    return performance

def print_performance_summary(performance):
    """Print summary of key performance metrics"""
    if not performance:
        print("No performance data available.")
        return
    
    print("\n=== Performance Summary ===")
    
    key_metrics = [
        "total_trades", "win_rate", "profit_factor", 
        "total_profit_usd", "max_drawdown_percent", 
        "sharpe_ratio", "sortino_ratio"
    ]
    
    for metric in key_metrics:
        value = performance.get(metric, "N/A")
        if isinstance(value, (int, float)):
            if metric.endswith("_percent") or metric == "win_rate":
                print(f"{metric}: {value:.2f}%")
            elif metric.endswith("_ratio"):
                print(f"{metric}: {value:.2f}")
            elif metric.endswith("_usd"):
                print(f"{metric}: ${value:.2f}")
            else:
                print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value}")

def main():
    """Main function that handles command-line arguments and runs the selected mode"""
    # Create top-level parser
    parser = argparse.ArgumentParser(description='XAUUSD Trading Bot')
    parser.add_argument('--initial-capital', type=float, default=1000.0,
                        help='Initial capital in USD')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Backtest mode
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--data', type=str, default='xau.csv',
                                help='Path to price data CSV file')
    backtest_parser.add_argument('--timeframe', type=str, default='5T',
                                help='Timeframe for analysis (e.g. 1T, 5T, 15T, 1H)')
    backtest_parser.add_argument('--params', type=str, default=None,
                                help='Path to JSON file with parameters')
    backtest_parser.add_argument('--output', type=str, default='results/reports/backtest_report.txt',
                                help='Path to save the results report')
    backtest_parser.add_argument('--no-plots', action='store_true',
                                help='Disable generating plots')
    
    # Optimization mode
    optimize_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    optimize_parser.add_argument('--data', type=str, default='xau.csv',
                               help='Path to price data CSV file')
    optimize_parser.add_argument('--timeframe', type=str, default='5T',
                               help='Timeframe for analysis')
    optimize_parser.add_argument('--param-ranges', type=str, default=None,
                               help='Path to JSON file with parameter ranges')
    optimize_parser.add_argument('--output-dir', type=str, default='results/optimization',
                               help='Directory to save optimization results')
    optimize_parser.add_argument('--metric', type=str, default='profit_factor',
                               help='Metric to optimize')
    optimize_parser.add_argument('--combinations', type=int, default=100,
                               help='Maximum number of parameter combinations to test')
    optimize_parser.add_argument('--walk-forward', action='store_true',
                               help='Use walk-forward validation')
    optimize_parser.add_argument('--train-test-split', type=float, default=0.7,
                               help='Train/test split ratio for walk-forward validation')
    
    # Validation mode
    validate_parser = subparsers.add_parser('validate', help='Validate optimized parameters')
    validate_parser.add_argument('--data', type=str, default='xau.csv',
                                help='Path to price data CSV file')
    validate_parser.add_argument('--timeframe', type=str, default='5T',
                                help='Timeframe for analysis')
    validate_parser.add_argument('--params', type=str, required=True,
                                help='Path to JSON file with optimized parameters')
    validate_parser.add_argument('--output', type=str, default='results/reports/validation_report.txt',
                                help='Path to save the validation report')
    validate_parser.add_argument('--no-plots', action='store_true',
                                help='Disable generating plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run selected mode
    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'optimize':
        run_optimization(args)
    elif args.mode == 'validate':
        validate_optimized_parameters(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()