"""
Reporting utilities for XAUUSD Trading Bot.

This module provides functions for generating performance reports
and detailed trade analysis.
"""

def generate_performance_report(bot):
    """
    Generate comprehensive trade report with detailed analysis
    
    Parameters:
    -----------
    bot : XAUUSDCentScalpingBot
        The bot instance with trades and performance data
        
    Returns:
    --------
    str
        Formatted report text
    """
    if not bot.trades:
        return "No trades to report."

    perf = bot.calculate_performance()
    report = "XAUUSD Cent-Account Scalping Bot — Trade Report\n" + "=" * 80 + "\n\n"
    
    # Performance summary section
    report += "Performance Summary\n" + "-" * 50 + "\n"
    for k, v in perf.items():
        if isinstance(v, dict) and k not in ["daily_results", "signal_counts", "duration_distribution"]:
            continue  # Handle these separately
        elif k.endswith("_usd"):
            report += f"{k:25s}: ${v:.2f}\n"
        elif k.endswith("_percent") or k == "win_rate":
            report += f"{k:25s}: {v:.2f}%\n"
        elif k in ["daily_results", "signal_counts", "duration_distribution"]:
            continue  # Handle these separately
        else:
            report += f"{k:25s}: {v}\n"
    report += "\n"
    
    # Duration distribution
    if "duration_distribution" in perf:
        report += "Trade Duration Distribution\n" + "-" * 50 + "\n"
        for duration, count in perf["duration_distribution"].items():
            report += f"{duration:15s}: {count} trades ({count/perf['total_trades']*100:.1f}% of total)\n"
        report += "\n"
    
    # Signal analysis
    if perf.get("signal_counts"):
        report += "Signal Analysis\n" + "-" * 50 + "\n"
        for sig, count in perf["signal_counts"].items():
            report += f"{sig:25s}: {count}\n"
        report += "\n"

    # Monthly performance
    if bot.daily_results:
        monthly = {}
        for date, profit in sorted(bot.daily_results.items()):
            key = date.strftime("%Y-%m")
            if key not in monthly:
                monthly[key] = 0
            monthly[key] += profit

        report += "Monthly Performance\n" + "-" * 50 + "\n"
        for month, profit in sorted(monthly.items()):
            report += f"{month:10s}: {profit/100:+.2f}$ {'✅' if profit > 0 else '❌'}\n"
        report += "\n"

    # Trade details table
    report += (
        "Trade Details\n"
        + "-" * 80
        + "\n"
        + f"{'#':<4}{'Type':<6}{'Entry':<20}{'Exit':<20}{'Dur(min)':<10}{'P/L($)':<10}{'Reason':<15}\n"
        + "-" * 80
        + "\n"
    )

    for i, t in enumerate(bot.trades, 1):
        dur = t.get("duration_minutes", 0)
        report += (
            f"{i:<4}{t['position']:<6}{t['entry_time']:%F %T} "
            f"{t['exit_time']:%F %T} {dur:<10.1f}"
            f"{t['profit']/100:+<10.2f}{t.get('exit_reason', ''):<15}\n"
        )

    # Add risk analysis section
    report += "\nRisk Analysis\n" + "-" * 50 + "\n"
    report += f"Initial capital: ${bot.initial_capital/100:.2f}\n"
    report += f"Final capital: ${perf.get('total_profit_cents', 0)/100 + bot.initial_capital/100:.2f}\n"
    report += f"Max drawdown: {perf.get('max_drawdown_percent', 0):.2f}%\n"
    report += f"Risk per trade: {bot.risk_percent:.2f}%\n"
    report += f"Profit factor: {perf.get('profit_factor', 0):.2f}\n"
    report += f"Sharpe ratio: {perf.get('sharpe_ratio', 0):.2f}\n"
    report += f"Sortino ratio: {perf.get('sortino_ratio', 0):.2f}\n"
    report += f"Win/Loss ratio: {perf.get('win_rate', 0):.2f}%\n"
    report += f"Average R multiple: {perf.get('average_r_multiple', 0):.2f}\n"
    report += f"Expectancy per trade: {perf.get('expectancy_per_trade', 0)/100:.2f}$\n"
    
    # Strategy parameters
    report += "\nStrategy Parameters\n" + "-" * 50 + "\n"
    report += f"Fast MA Period: {bot.fast_ma_period}\n"
    report += f"Slow MA Period: {bot.slow_ma_period}\n"
    report += f"Stochastic K Period: {bot.stoch_k_period}\n"
    report += f"Stochastic D Period: {bot.stoch_d_period}\n"
    report += f"Stochastic Upper Level: {bot.stoch_upper_level}\n"
    report += f"Stochastic Lower Level: {bot.stoch_lower_level}\n"
    
    if bot.use_rsi_filter:
        report += f"RSI Period: {bot.rsi_period}\n"
        report += f"RSI Overbought: {bot.rsi_overbought}\n"
        report += f"RSI Oversold: {bot.rsi_oversold}\n"
    
    if bot.use_adx_filter:
        report += f"ADX Period: {bot.adx_period}\n"
        report += f"ADX Threshold: {bot.adx_threshold}\n"
    
    if bot.use_atr_filter:
        report += f"ATR Period: {bot.atr_period}\n"
        report += f"ATR Multiplier: {bot.atr_multiplier}\n"
    
    report += f"Take Profit: {bot.take_profit_points} points\n"
    report += f"Stop Loss: {bot.stop_loss_points} points\n"
    report += f"Trailing Stop: {'Enabled' if bot.trailing_stop else 'Disabled'}\n"
    
    if bot.trailing_stop:
        report += f"Trailing Distance: {bot.trailing_distance} points\n"
    
    if bot.use_partial_take_profit:
        report += f"Partial Take Profit: {bot.partial_tp_ratio * 100}% at {bot.partial_tp_distance * 100}% of target\n"
    
    if bot.use_breakeven_stop:
        report += f"Breakeven Stop: After {bot.breakeven_trigger_ratio * 100}% of target reached\n"
    
    report += f"Trading Hours: {bot.trading_hours['start']}:00 - {bot.trading_hours['end']}:00\n"
    report += f"Max Daily Loss: {bot.max_daily_loss}%\n"
    
    return report

def generate_optimization_report(results, output_file=None):
    """
    Generate report on optimization results
    
    Parameters:
    -----------
    results : list
        List of optimization results
    output_file : str, optional
        File to save the report to
        
    Returns:
    --------
    str
        Formatted report text
    """
    if not results:
        return "No optimization results to report."
        
    report = "XAUUSD Bot Optimization Results\n" + "=" * 80 + "\n\n"
    
    # Top parameters section
    report += f"Total Parameter Combinations Tested: {len(results)}\n\n"
    report += "Top 10 Parameter Sets\n" + "-" * 50 + "\n"
    
    # Determine if walk-forward was used
    is_walk_forward = "train_metrics" in results[0] if results else False
    
    # Display top results
    for i, r in enumerate(results[:10], 1):
        if is_walk_forward:
            report += f"#{i} - Combined Score: {r.get('combined_score', 0):.2f}\n"
            report += f"  Training: PF={r['train_metrics']['profit_factor']:.2f}, "
            report += f"Win={r['train_metrics']['win_rate']:.1f}%, "
            report += f"P&L=${r['train_metrics']['total_profit_usd']:.2f}\n"
            
            report += f"  Testing:  PF={r['test_metrics'].get('profit_factor', 0):.2f}, "
            report += f"Win={r['test_metrics'].get('win_rate', 0):.1f}%, "
            report += f"P&L=${r['test_metrics'].get('total_profit_usd', 0):.2f}\n"
            
            report += f"  Robustness: {r.get('robustness', 0):.2f}\n"
        else:
            metrics = r["metrics"]
            report += f"#{i} - Score: {r.get('score', 0):.2f}\n"
            report += f"  PF={metrics['profit_factor']:.2f}, "
            report += f"Win={metrics['win_rate']:.1f}%, "
            report += f"P&L=${metrics['total_profit_usd']:.2f}, "
            report += f"DD={metrics['max_drawdown_percent']:.1f}%, "
            report += f"Trades={metrics['total_trades']}\n"
        
        # Parameter values
        report += "  Parameters:\n"
        for param, value in r["parameters"].items():
            report += f"    {param}: {value}\n"
        report += "\n"
    
    # Parameter frequency analysis
    if len(results) >= 5:
        report += "Parameter Frequency Analysis (Top 5 Results)\n" + "-" * 50 + "\n"
        
        # Get all parameters
        all_params = results[0]["parameters"].keys()
        
        for param in all_params:
            param_values = [r["parameters"][param] for r in results[:5]]
            value_counts = {}
            
            for value in param_values:
                if value not in value_counts:
                    value_counts[value] = 0
                value_counts[value] += 1
            
            # Sort by frequency
            sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            
            report += f"{param}:\n"
            for value, count in sorted_values:
                report += f"  {value}: {count} occurrences ({count/5*100:.0f}%)\n"
            report += "\n"
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Optimization report saved to {output_file}")
    
    return report