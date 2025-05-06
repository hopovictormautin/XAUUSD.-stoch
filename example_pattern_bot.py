"""
Example script demonstrating the usage of the enhanced pattern-based XAUUSD Trading Bot.

This script runs a backtest with the pattern-enhanced bot and compares its
performance against the standard bot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import DEFAULT_PARAMS
from xauusd_bot import XAUUSDCentScalpingBot
from xauusd_pattern_bot import XAUUSDPatternBot
from pattern_integration import integrate_patterns_with_bot

def main():
    """Run a comparative backtest between standard and pattern-enhanced bots"""
    # Create results directory
    os.makedirs('results/reports', exist_ok=True)
    
    # Load price data
    csv_file = 'xau.csv'
    timeframe = '5T'
    
    print("Running comparative backtest on XAUUSD cent account scalping bot...")
    print("Comparing standard strategy vs pattern-enhanced strategy")
    print("=" * 70)
    
    # ---- Create standard bot with default parameters ----
    standard_bot = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    
    # ---- Create pattern-enhanced bot ----
    # Method 1: Using dedicated pattern bot class
    pattern_bot = XAUUSDPatternBot(
        **DEFAULT_PARAMS,
        use_pattern_recognition=True,
        pattern_position_sizing=True,
        risk_averse_entry=False,
        min_body_pct=0.2,
        max_body_pct=15.0
    )
    
    # Method 2: Enhancing standard bot with pattern recognition
    # (Uncomment to use this method instead)
    # standard_bot_copy = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    # enhanced_bot = integrate_patterns_with_bot(standard_bot_copy)
    
    # ---- Run backtests ----
    print("\nRunning standard bot backtest...")
    standard_performance = standard_bot.backtest(csv_file, use_timeframe=timeframe)
    
    print("\nRunning pattern-enhanced bot backtest...")
    pattern_performance = pattern_bot.backtest(csv_file, use_timeframe=timeframe)
    
    # ---- Generate reports ----
    if standard_performance and pattern_performance:
        standard_bot.generate_trade_report("results/reports/standard_strategy_report.txt")
        pattern_bot.generate_trade_report("results/reports/pattern_strategy_report.txt")
        
        # Create comparative summary
        print("\n" + "=" * 30 + " COMPARISON " + "=" * 30)
        metrics = [
            "total_trades", "win_rate", "profit_factor", 
            "total_profit_usd", "max_drawdown_percent", 
            "sharpe_ratio", "sortino_ratio"
        ]
        
        print(f"{'Metric':<25}{'Standard Bot':<15}{'Pattern Bot':<15}{'Difference':<15}")
        print("-" * 70)
        
        for metric in metrics:
            std_val = standard_performance.get(metric, 0)
            pat_val = pattern_performance.get(metric, 0)
            diff = pat_val - std_val
            diff_pct = (diff / std_val * 100) if std_val != 0 else float('inf')
            
            # Format values based on metric type
            if metric.endswith("_percent") or metric == "win_rate":
                std_str = f"{std_val:.2f}%"
                pat_str = f"{pat_val:.2f}%"
                diff_str = f"{diff:.2f}% ({diff_pct:+.1f}%)"
            elif metric.endswith("_ratio"):
                std_str = f"{std_val:.2f}"
                pat_str = f"{pat_val:.2f}"
                diff_str = f"{diff:.2f} ({diff_pct:+.1f}%)"
            elif metric.endswith("_usd"):
                std_str = f"${std_val:.2f}"
                pat_str = f"${pat_val:.2f}"
                diff_str = f"${diff:.2f} ({diff_pct:+.1f}%)"
            else:
                std_str = f"{std_val}"
                pat_str = f"{pat_val}"
                diff_str = f"{diff} ({diff_pct:+.1f}%)"
                
            print(f"{metric:<25}{std_str:<15}{pat_str:<15}{diff_str:<15}")
        
        # ---- Generate comparative equity curve ----
        plt.figure(figsize=(12, 6))
        
        # Convert to dollars
        standard_equity = [bal/100 for bal in standard_bot.equity_curve]
        pattern_equity = [bal/100 for bal in pattern_bot.equity_curve]
        
        # Ensure both arrays are the same length for plotting
        min_len = min(len(standard_equity), len(pattern_equity))
        
        plt.plot(standard_equity[:min_len], label='Standard Strategy', color='blue', alpha=0.7)
        plt.plot(pattern_equity[:min_len], label='Pattern-Enhanced Strategy', color='green', alpha=0.7)
        
        plt.title("Equity Curve Comparison ($)")
        plt.ylabel("Balance ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and show plot
        plt.savefig("results/charts/strategy_comparison.png", dpi=300)
        plt.show()
        
        # Report pattern-specific statistics if available
        if 'pattern_stats' in pattern_performance:
            print("\n" + "=" * 25 + " PATTERN STATISTICS " + "=" * 25)
            for pattern, stats in pattern_performance['pattern_stats'].items():
                if stats['count'] > 0:
                    win_rate = (stats['wins'] / (stats['wins'] + stats['losses'])) * 100 if stats['wins'] + stats['losses'] > 0 else 0
                    print(f"{pattern:<20}: {stats['count']} trades, {win_rate:.1f}% win rate ({stats['wins']}/{stats['wins'] + stats['losses']})")
    else:
        print("One or both backtests did not produce results. Check for errors.")

if __name__ == "__main__":
    main()