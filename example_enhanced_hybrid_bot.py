"""
Example script demonstrating the usage of the enhanced hybrid XAUUSD Trading Bot.

This script runs a backtest with the optimized hybrid bot and compares its
performance against the standard bot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import DEFAULT_PARAMS
from xauusd_bot import XAUUSDCentScalpingBot
from xauusd_hybrid_bot_enhanced import XAUUSDHybridBot

def main():
    """Run a comparative backtest between standard and enhanced hybrid bot"""
    # Create results directory
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('results/charts', exist_ok=True)
    
    # Load price data
    csv_file = 'xau.csv'
    timeframe = '5T'
    
    print("Running comparative backtest on XAUUSD cent account scalping bot...")
    print("Comparing standard strategy vs enhanced hybrid strategy")
    print("=" * 70)
    
    # ---- Create standard bot with default parameters ----
    standard_bot = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    
    # ---- Create Enhanced Hybrid bot with optimized parameters ----
    hybrid_bot = XAUUSDHybridBot(
        **DEFAULT_PARAMS,
        # Core Fibonacci parameters
        use_fibonacci=True,
        fibonacci_lookback=35,      # More responsive to recent price action
        swing_lookback=8,           # Identifies more swing points
        fib_trend_window=15,        # More sensitive trend detection
        
        # Fibonacci level preferences
        prioritize_level_382=True,  # Focus on 0.382 level
        prioritize_level_618=True,  # Add focus on 0.618 level (golden ratio)
        
        # Signal processing
        fibonacci_filter_only=True, # Use Fibonacci for filtering
        trend_confirmation=True,    # Only take trades in direction of overall trend
        
        # Enhanced exit management
        enhanced_exits=True,        # Use Fibonacci-based exits
        use_multiple_targets=True,  # Use multiple take profit targets
        trailing_exit_factor=1.5    # Dynamic trailing stop based on ATR
    )
    
    # ---- Run backtests ----
    print("\nRunning standard bot backtest...")
    standard_performance = standard_bot.backtest(csv_file, use_timeframe=timeframe)
    
    print("\nRunning enhanced hybrid bot backtest...")
    hybrid_performance = hybrid_bot.backtest(csv_file, use_timeframe=timeframe)
    
    # ---- Generate reports ----
    if standard_performance and hybrid_performance:
        standard_bot.generate_trade_report("results/reports/standard_strategy_report.txt")
        hybrid_bot.generate_trade_report("results/reports/enhanced_hybrid_report.txt")
        
        # Create comparative summary
        print("\n" + "=" * 30 + " COMPARISON " + "=" * 30)
        metrics = [
            "total_trades", "win_rate", "profit_factor", 
            "total_profit_usd", "max_drawdown_percent", 
            "sharpe_ratio", "sortino_ratio"
        ]
        
        print(f"{'Metric':<25}{'Standard':<15}{'Enhanced Hybrid':<20}{'Difference':<15}")
        print("-" * 75)
        
        for metric in metrics:
            std_val = standard_performance.get(metric, 0)
            hyb_val = hybrid_performance.get(metric, 0)
            
            # Calculate difference vs standard
            hyb_diff = hyb_val - std_val
            hyb_diff_pct = (hyb_diff / std_val * 100) if std_val != 0 else float('inf')
            
            # Format values based on metric type
            if metric.endswith("_percent") or metric == "win_rate":
                std_str = f"{std_val:.2f}%"
                hyb_str = f"{hyb_val:.2f}%"
                diff_str = f"{hyb_diff:+.2f}% ({hyb_diff_pct:+.1f}%)"
            elif metric.endswith("_ratio"):
                std_str = f"{std_val:.2f}"
                hyb_str = f"{hyb_val:.2f}"
                diff_str = f"{hyb_diff:+.2f} ({hyb_diff_pct:+.1f}%)"
            elif metric.endswith("_usd"):
                std_str = f"${std_val:.2f}"
                hyb_str = f"${hyb_val:.2f}"
                diff_str = f"${hyb_diff:+.2f} ({hyb_diff_pct:+.1f}%)"
            else:
                std_str = f"{std_val}"
                hyb_str = f"{hyb_val}"
                diff_str = f"{hyb_diff:+.0f} ({hyb_diff_pct:+.1f}%)"
                
            print(f"{metric:<25}{std_str:<15}{hyb_str:<20}{diff_str:<15}")
        
        # ---- Generate comparative equity curve ----
        plt.figure(figsize=(12, 6))
        
        # Convert to dollars
        standard_equity = [bal/100 for bal in standard_bot.equity_curve]
        hybrid_equity = [bal/100 for bal in hybrid_bot.equity_curve]
        
        # Ensure all arrays are the same length for plotting
        min_len = min(len(standard_equity), len(hybrid_equity))
        
        plt.plot(standard_equity[:min_len], label='Standard Strategy', color='blue', alpha=0.7)
        plt.plot(hybrid_equity[:min_len], label='Enhanced Hybrid Strategy', color='red', alpha=0.7)
        
        plt.title("Equity Curve Comparison ($)")
        plt.ylabel("Balance ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and show plot
        plt.savefig("results/charts/enhanced_comparison.png", dpi=300)
        plt.show()
        
        # Generate Fibonacci analysis chart if available
        fib_chart = hybrid_bot.plot_fibonacci_analysis(lookback=50)
        if fib_chart:
            fib_chart.savefig("results/charts/hybrid_fibonacci_analysis.png", dpi=300)
            plt.show()
            
        # ---- Print enhanced hybrid strategy details ----
        if 'enhanced_entry_pct' in hybrid_performance and 'enhanced_exit_pct' in hybrid_performance:
            print("\n" + "=" * 25 + " ENHANCED HYBRID STRATEGY DETAILS " + "=" * 25)
            print(f"Enhanced entries: {hybrid_performance['enhanced_entries']} ({hybrid_performance['enhanced_entry_pct']:.1f}%)")
            print(f"Enhanced exits: {hybrid_performance['enhanced_exits']} ({hybrid_performance['enhanced_exit_pct']:.1f}%)")
            print(f"Fibonacci filtered signals: {hybrid_performance['filtered_signals']}")
            print(f"Trend filtered signals: {hybrid_performance['trend_filtered_signals']}")
            
            if 'partial_exits_count' in hybrid_performance:
                print(f"Partial exits: {hybrid_performance['partial_exits_count']} (${hybrid_performance['partial_exits_profit']:.2f} profit)")
            
            if 'trailing_stops_activated' in hybrid_performance:
                print(f"Trailing stops activated: {hybrid_performance['trailing_stops_activated']}")
            
            # Display Fibonacci level performance
            print("\nFibonacci Level Performance:")
            if 'fibonacci_stats' in hybrid_performance:
                for level, stats in hybrid_performance['fibonacci_stats'].items():
                    if stats['count'] > 0:
                        wins = stats['wins']
                        total_results = stats['wins'] + stats['losses']
                        win_rate = (wins / total_results) * 100 if total_results > 0 else 0
                        print(f"Level {level}: {win_rate:.1f}% win rate ({stats['count']} trades, {wins}/{total_results} completed)")
    else:
        print("One or more backtests did not produce results. Check for errors.")

if __name__ == "__main__":
    main()