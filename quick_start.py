"""
Quick start script for XAUUSD Trading Bot.

This script provides a simple way to run a backtest with the improved strategy.
"""

import os
from xauusd_bot import XAUUSDCentScalpingBot
from config import DEFAULT_PARAMS


def main():
    """Run a quick backtest with the improved strategy parameters"""
    # Create results directory
    os.makedirs('results/reports', exist_ok=True)
    

     # Log the parameters being used
    print("\n=== STRATEGY PARAMETERS BEING USED ===")
    for key, value in DEFAULT_PARAMS.items():
        print(f"{key}: {value}")
    print("========================================\n")
    
    # Create bot with improved parameters from config
    bot = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    
    # Run backtest on 5-minute timeframe
    print("Running backtest on 5-minute timeframe with optimized parameters...")
    performance = bot.backtest('xau.csv', use_timeframe='5T')
    
    if performance:
        # Generate visualization
        bot.plot_results()
        
        # Save detailed report
        bot.generate_trade_report("results/reports/improved_strategy_report.txt")
        
        # Print summary
        print("\n=== Performance Summary ===")
        key_metrics = [
            "total_trades", "win_rate", "profit_factor", 
            "total_profit_usd", "max_drawdown_percent", 
            "sharpe_ratio", "sortino_ratio", "expectancy_per_trade"
        ]
        
        for metric in key_metrics:
            value = performance.get(metric, "N/A")
            if isinstance(value, (int, float)):
                if metric.endswith("_percent") or metric == "win_rate":
                    print(f"{metric}: {value:.2f}%")
                elif metric.endswith("_ratio"):
                    print(f"{metric}: {value:.2f}")
                elif metric.endswith("_usd") or metric == "expectancy_per_trade":
                    print(f"{metric}: ${value:.2f}")
                else:
                    print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value}")
    else:
        print("Backtest did not produce results. Check for errors.")

if __name__ == "__main__":
    main()