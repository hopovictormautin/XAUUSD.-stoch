"""
Script to run backtests on the XAUUSD Trading Bot.

This script provides a command-line interface for running backtests
with the XAUUSD cent account scalping bot.
"""

import argparse
import time
import os
import json
from xauusd_bot import XAUUSDCentScalpingBot

def main():
    """Main function to run bot backtests"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run XAUUSD Trading Bot Backtest')
    
    parser.add_argument('--data', type=str, default='xau.csv',
                       help='Path to price data CSV file')
    parser.add_argument('--timeframe', type=str, default='5T',
                       help='Timeframe for analysis (e.g. 1T, 5T, 15T, 1H)')
    parser.add_argument('--params', type=str, default=None,
                       help='Path to JSON file with parameters')
    parser.add_argument('--output', type=str, default='results/reports/xauusd_backtest_report.txt',
                       help='Path to save the results report')
    parser.add_argument('--initial-capital', type=float, default=1000.0,
                       help='Initial capital in USD')
    parser.add_argument('--disable-plots', action='store_true',
                       help='Disable generating plots')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load parameters from file if provided
    if args.params:
        try:
            with open(args.params, 'r') as f:
                bot_params = json.load(f)
            print(f"Loaded parameters from {args.params}")
        except Exception as e:
            print(f"Error loading parameters file: {e}")
            print("Using default parameters instead.")
            bot_params = {}
    else:
        bot_params = {}
    
    # Add initial capital converted to cents
    bot_params['initial_capital'] = int(args.initial_capital * 100)
    
    # Create bot instance
    bot = XAUUSDCentScalpingBot(**bot_params)
    
    # Run backtest
    print(f"Running backtest on {args.timeframe} timeframe using data from: {args.data}")
    start_time = time.time()
    
    performance = bot.backtest(args.data, use_timeframe=args.timeframe)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Backtest completed in {duration:.2f} seconds")
    
    # Generate plots if not disabled
    if not args.disable_plots:
        bot.plot_results()
    
    # Generate and save report
    if performance:
        bot.generate_trade_report(args.output)
        print(f"Report saved to {args.output}")
        
        # Print key metrics summary
        print("\n=== Performance Summary ===")
        key_metrics = [
            "total_trades", "win_rate", "profit_factor", 
            "total_profit_usd", "max_drawdown_percent", 
            "sharpe_ratio"
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
    else:
        print("No performance results generated. Check for errors.")

if __name__ == "__main__":
    main()