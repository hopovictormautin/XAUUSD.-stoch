"""
Example script demonstrating the usage of the hybrid XAUUSD Trading Bot.

This script runs a backtest with the hybrid bot and compares its
performance against both the standard bot and Fibonacci-only bot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import DEFAULT_PARAMS
from xauusd_bot import XAUUSDCentScalpingBot
from xauusd_fibonacci_bot import XAUUSDFibonacciBot
from xauusd_hybrid_bot import XAUUSDHybridBot

def main():
    """Run a comparative backtest between different bot implementations"""
    # Create results directory
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('results/charts', exist_ok=True)
    
    # Load price data
    csv_file = 'xau.csv'
    timeframe = '5T'
    
    print("Running comparative backtest on XAUUSD cent account scalping bot...")
    print("Comparing standard vs Fibonacci vs Hybrid strategy")
    print("=" * 70)
    
    # ---- Create standard bot with default parameters ----
    standard_bot = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    
    # ---- Create Fibonacci-only bot ----
    fibonacci_bot = XAUUSDFibonacciBot(
        **DEFAULT_PARAMS,
        use_fibonacci=True,
        fibonacci_lookback=50,
        swing_lookback=10,
        fib_trend_window=20,
        prefer_fibonacci_stops=True
    )
    
    # ---- Create Hybrid bot ----
    # This combines standard strategy with Fibonacci enhancements
    hybrid_bot = XAUUSDHybridBot(
        **DEFAULT_PARAMS,
        use_fibonacci=True,
        fibonacci_lookback=50,
        swing_lookback=10,
        fib_trend_window=20,
        prioritize_level_382=True,
        fibonacci_filter_only=True,  # Use Fibonacci only for filtering, not generating signals
        enhanced_exits=True
    )
    
    # ---- Run backtests ----
    print("\nRunning standard bot backtest...")
    standard_performance = standard_bot.backtest(csv_file, use_timeframe=timeframe)
    
    print("\nRunning Fibonacci-only bot backtest...")
    fibonacci_performance = fibonacci_bot.backtest(csv_file, use_timeframe=timeframe)
    
    print("\nRunning Hybrid bot backtest...")
    hybrid_performance = hybrid_bot.backtest(csv_file, use_timeframe=timeframe)
    
    # ---- Generate reports ----
    if standard_performance and hybrid_performance:
        standard_bot.generate_trade_report("results/reports/standard_strategy_report.txt")
        fibonacci_bot.generate_trade_report("results/reports/fibonacci_strategy_report.txt")
        hybrid_bot.generate_trade_report("results/reports/hybrid_strategy_report.txt")
        
        # Create comparative summary
        print("\n" + "=" * 30 + " COMPARISON " + "=" * 30)
        metrics = [
            "total_trades", "win_rate", "profit_factor", 
            "total_profit_usd", "max_drawdown_percent", 
            "sharpe_ratio", "sortino_ratio"
        ]
        
        print(f"{'Metric':<25}{'Standard':<15}{'Fibonacci':<15}{'Hybrid':<15}")
        print("-" * 70)
        
        for metric in metrics:
            std_val = standard_performance.get(metric, 0)
            fib_val = fibonacci_performance.get(metric, 0)
            hyb_val = hybrid_performance.get(metric, 0)
            
            # Format values based on metric type
            if metric.endswith("_percent") or metric == "win_rate":
                std_str = f"{std_val:.2f}%"
                fib_str = f"{fib_val:.2f}%"
                hyb_str = f"{hyb_val:.2f}%"
                
                # Calculate difference vs standard
                hyb_diff = hyb_val - std_val
                hyb_diff_pct = (hyb_diff / std_val * 100) if std_val != 0 else float('inf')
                hyb_diff_str = f"{hyb_diff:+.2f}% ({hyb_diff_pct:+.1f}%)"
            elif metric.endswith("_ratio"):
                std_str = f"{std_val:.2f}"
                fib_str = f"{fib_val:.2f}"
                hyb_str = f"{hyb_val:.2f}"
                
                # Calculate difference vs standard
                hyb_diff = hyb_val - std_val
                hyb_diff_pct = (hyb_diff / std_val * 100) if std_val != 0 else float('inf')
                hyb_diff_str = f"{hyb_diff:+.2f} ({hyb_diff_pct:+.1f}%)"
            elif metric.endswith("_usd"):
                std_str = f"${std_val:.2f}"
                fib_str = f"${fib_val:.2f}"
                hyb_str = f"${hyb_val:.2f}"
                
                # Calculate difference vs standard
                hyb_diff = hyb_val - std_val
                hyb_diff_pct = (hyb_diff / std_val * 100) if std_val != 0 else float('inf')
                hyb_diff_str = f"${hyb_diff:+.2f} ({hyb_diff_pct:+.1f}%)"
            else:
                std_str = f"{std_val}"
                fib_str = f"{fib_val}"
                hyb_str = f"{hyb_val}"
                
                # Calculate difference vs standard
                hyb_diff = hyb_val - std_val
                hyb_diff_pct = (hyb_diff / std_val * 100) if std_val != 0 else float('inf')
                hyb_diff_str = f"{hyb_diff:+.0f} ({hyb_diff_pct:+.1f}%)"
                
            print(f"{metric:<25}{std_str:<15}{fib_str:<15}{hyb_str:<15} {hyb_diff_str}")
        
        # ---- Generate comparative equity curve ----
        plt.figure(figsize=(12, 6))
        
        # Convert to dollars
        standard_equity = [bal/100 for bal in standard_bot.equity_curve]
        fibonacci_equity = [bal/100 for bal in fibonacci_bot.equity_curve]
        hybrid_equity = [bal/100 for bal in hybrid_bot.equity_curve]
        
        # Ensure all arrays are the same length for plotting
        min_len = min(len(standard_equity), len(fibonacci_equity), len(hybrid_equity))
        
        plt.plot(standard_equity[:min_len], label='Standard Strategy', color='blue', alpha=0.7)
        plt.plot(fibonacci_equity[:min_len], label='Fibonacci Strategy', color='green', alpha=0.5)
        plt.plot(hybrid_equity[:min_len], label='Hybrid Strategy', color='red', alpha=0.7)
        
        plt.title("Equity Curve Comparison ($)")
        plt.ylabel("Balance ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and show plot
        plt.savefig("results/charts/strategy_comparison.png", dpi=300)
        plt.show()
        
        # ---- Print hybrid strategy details ----
        if 'enhanced_entry_pct' in hybrid_performance and 'enhanced_exit_pct' in hybrid_performance:
            print("\n" + "=" * 25 + " HYBRID STRATEGY DETAILS " + "=" * 25)
            print(f"Enhanced entries: {hybrid_performance['enhanced_entries']} ({hybrid_performance['enhanced_entry_pct']:.1f}%)")
            print(f"Enhanced exits: {hybrid_performance['enhanced_exits']} ({hybrid_performance['enhanced_exit_pct']:.1f}%)")
            print(f"Filtered signals: {hybrid_performance['filtered_signals']}")
            
            # Generate Fibonacci analysis chart if available
            fib_chart = hybrid_bot.plot_fibonacci_analysis(lookback=50)
            if fib_chart:
                fib_chart.savefig("results/charts/hybrid_fibonacci_analysis.png", dpi=300)
                plt.show()
    else:
        print("One or more backtests did not produce results. Check for errors.")

if __name__ == "__main__":
    main()