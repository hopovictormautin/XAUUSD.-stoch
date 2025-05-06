"""
Example script demonstrating the usage of the enhanced Fibonacci-based XAUUSD Trading Bot.

This script runs a backtest with the Fibonacci-enhanced bot and compares its
performance against the standard bot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import DEFAULT_PARAMS
from xauusd_bot import XAUUSDCentScalpingBot
from xauusd_fibonacci_bot import XAUUSDFibonacciBot

def main():
    """Run a comparative backtest between standard and Fibonacci-enhanced bots"""
    # Create results directory
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('results/charts', exist_ok=True)
    
    # Load price data
    csv_file = 'xau.csv'
    timeframe = '5T'
    
    print("Running comparative backtest on XAUUSD cent account scalping bot...")
    print("Comparing standard strategy vs Fibonacci-enhanced strategy")
    print("=" * 70)
    
    # ---- Create standard bot with default parameters ----
    standard_bot = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    
    # ---- Create Fibonacci-enhanced bot ----
    fibonacci_bot = XAUUSDFibonacciBot(
        **DEFAULT_PARAMS,
        use_fibonacci=True,
        fibonacci_lookback=50,
        swing_lookback=10,
        fib_trend_window=20,
        prefer_fibonacci_stops=True
    )
    
    # ---- Run backtests ----
    print("\nRunning standard bot backtest...")
    standard_performance = standard_bot.backtest(csv_file, use_timeframe=timeframe)
    
    print("\nRunning Fibonacci-enhanced bot backtest...")
    fibonacci_performance = fibonacci_bot.backtest(csv_file, use_timeframe=timeframe)
    
    # ---- Generate reports ----
    if standard_performance and fibonacci_performance:
        standard_bot.generate_trade_report("results/reports/standard_strategy_report.txt")
        fibonacci_bot.generate_trade_report("results/reports/fibonacci_strategy_report.txt")
        
        # Create comparative summary
        print("\n" + "=" * 30 + " COMPARISON " + "=" * 30)
        metrics = [
            "total_trades", "win_rate", "profit_factor", 
            "total_profit_usd", "max_drawdown_percent", 
            "sharpe_ratio", "sortino_ratio"
        ]
        
        print(f"{'Metric':<25}{'Standard Bot':<15}{'Fibonacci Bot':<15}{'Difference':<15}")
        print("-" * 70)
        
        for metric in metrics:
            std_val = standard_performance.get(metric, 0)
            fib_val = fibonacci_performance.get(metric, 0)
            diff = fib_val - std_val
            diff_pct = (diff / std_val * 100) if std_val != 0 else float('inf')
            
            # Format values based on metric type
            if metric.endswith("_percent") or metric == "win_rate":
                std_str = f"{std_val:.2f}%"
                fib_str = f"{fib_val:.2f}%"
                diff_str = f"{diff:.2f}% ({diff_pct:+.1f}%)"
            elif metric.endswith("_ratio"):
                std_str = f"{std_val:.2f}"
                fib_str = f"{fib_val:.2f}"
                diff_str = f"{diff:.2f} ({diff_pct:+.1f}%)"
            elif metric.endswith("_usd"):
                std_str = f"${std_val:.2f}"
                fib_str = f"${fib_val:.2f}"
                diff_str = f"${diff:.2f} ({diff_pct:+.1f}%)"
            else:
                std_str = f"{std_val}"
                fib_str = f"{fib_val}"
                diff_str = f"{diff} ({diff_pct:+.1f}%)"
                
            print(f"{metric:<25}{std_str:<15}{fib_str:<15}{diff_str:<15}")
        
        # ---- Generate comparative equity curve ----
        plt.figure(figsize=(12, 6))
        
        # Convert to dollars
        standard_equity = [bal/100 for bal in standard_bot.equity_curve]
        fibonacci_equity = [bal/100 for bal in fibonacci_bot.equity_curve]
        
        # Ensure both arrays are the same length for plotting
        min_len = min(len(standard_equity), len(fibonacci_equity))
        
        plt.plot(standard_equity[:min_len], label='Standard Strategy', color='blue', alpha=0.7)
        plt.plot(fibonacci_equity[:min_len], label='Fibonacci Strategy', color='green', alpha=0.7)
        
        plt.title("Equity Curve Comparison ($)")
        plt.ylabel("Balance ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and show plot
        plt.savefig("results/charts/strategy_comparison.png", dpi=300)
        plt.show()
        
        # Generate Fibonacci analysis chart if available
        fib_chart = fibonacci_bot.plot_fibonacci_analysis(lookback=50)
        if fib_chart:
            fib_chart.savefig("results/charts/fibonacci_analysis.png", dpi=300)
            plt.show()
        
        # Report Fibonacci-specific statistics if available
        if 'fibonacci_stats' in fibonacci_performance:
            print("\n" + "=" * 25 + " FIBONACCI STATISTICS " + "=" * 25)
            
            # Print confluence information
            if 'confluence_percentage' in fibonacci_performance:
                print(f"Confluence Percentage: {fibonacci_performance['confluence_percentage']:.1f}%")
                
            # Print level performance
            print("\nPerformance by Fibonacci level:")
            for level, stats in fibonacci_performance['fibonacci_stats'].items():
                if stats['count'] > 0:
                    wins = stats['wins']
                    total_results = stats['wins'] + stats['losses']
                    win_rate = (wins / total_results) * 100 if total_results > 0 else 0
                    print(f"Level {level:<8}: {stats['count']:3d} trades, {win_rate:5.1f}% win rate ({wins}/{total_results} completed)")
    else:
        print("One or both backtests did not produce results. Check for errors.")

if __name__ == "__main__":
    main()