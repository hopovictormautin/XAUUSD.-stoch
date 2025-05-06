"""
Enhanced XAUUSD Trading Bot with Fibonacci Retracement

This module extends the base XAUUSDCentScalpingBot by adding
Fibonacci retracement analysis to improve entry and exit points.
"""

from xauusd_bot import XAUUSDCentScalpingBot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.fibonacci import generate_fibonacci_signals, plot_fibonacci_levels

class XAUUSDFibonacciBot(XAUUSDCentScalpingBot):
    """
    Enhanced trading bot that combines the original MA/Stochastic strategy
    with Fibonacci retracement analysis for improved signal quality.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Fibonacci-enhanced trading bot
        
        Parameters:
        -----------
        **kwargs : dict
            Bot parameters, passed to parent class
            
        Additional parameters:
        ---------------------
        use_fibonacci : bool
            Enable/disable Fibonacci analysis (default: True)
        fibonacci_lookback : int
            Bars to look back for Fibonacci analysis (default: 50)
        swing_lookback : int
            Bars to look for swing points (default: 10)
        fib_trend_window : int
            Window for trend detection (default: 20)
        prefer_fibonacci_stops : bool
            Whether to use Fibonacci-based stops (default: True)
        """
        # Extract Fibonacci-specific parameters with defaults
        self.use_fibonacci = kwargs.pop('use_fibonacci', True)
        self.fibonacci_lookback = kwargs.pop('fibonacci_lookback', 50)
        self.swing_lookback = kwargs.pop('swing_lookback', 10)
        self.fib_trend_window = kwargs.pop('fib_trend_window', 20)
        self.prefer_fibonacci_stops = kwargs.pop('prefer_fibonacci_stops', True)
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Add Fibonacci-specific attributes
        self.fibonacci_levels = {}
        self.fibonacci_signals = pd.DataFrame()
        
        # Tracking for Fibonacci stats
        self.fib_stats = {
            '0.236': {'count': 0, 'wins': 0, 'losses': 0},
            '0.382': {'count': 0, 'wins': 0, 'losses': 0},
            '0.5': {'count': 0, 'wins': 0, 'losses': 0},
            '0.618': {'count': 0, 'wins': 0, 'losses': 0},
            '0.786': {'count': 0, 'wins': 0, 'losses': 0}
        }
        
        # Record confluence patterns
        self.confluence_count = 0
        self.total_signals = 0
        
    def prepare_data(self, df):
        """
        Prepare data with indicators and Fibonacci analysis
        
        Extends the parent method by adding Fibonacci pattern identification
        """
        # Use parent method first to get technical indicators
        result = super().prepare_data(df)
        
        if self.use_fibonacci:
            print("\nFibonacci Analysis:")
            print(f"Input data shape: {result.shape}")
            
            # Generate Fibonacci signals
            fib_df = generate_fibonacci_signals(
                result,
                lookback=self.fibonacci_lookback,
                swing_lookback=self.swing_lookback,
                trend_window=self.fib_trend_window
            )
            
            # Count Fibonacci levels identified
            level_counts = {}
            if 'fib_level_touched' in fib_df.columns:
                level_counts = fib_df['fib_level_touched'].value_counts().to_dict()
                
            print("\nFibonacci levels identified:")
            for level, count in level_counts.items():
                if level is not None:
                    print(f"Level {level:.3f}: {count} instances")
            
            # Count signals generated
            signal_counts = fib_df['fib_signal'].value_counts()
            print("\nFibonacci signal counts:")
            print(signal_counts)
            
            # Store Fibonacci data
            self.fibonacci_signals = fib_df[
                [
                    'fib_retracement', 'fib_level_touched',
                    'fib_start_price', 'fib_end_price',
                    'fib_signal', 'fib_stop_loss',
                    'fib_take_profit1', 'fib_take_profit2', 'fib_take_profit3'
                ]
            ]
            
            # Merge Fibonacci columns into result
            for col in self.fibonacci_signals.columns:
                result[col] = self.fibonacci_signals[col]
        
        return result
    
    def check_entry_signals(self, row):
        """
        Check for entry signals with both indicators and Fibonacci retracements
        
        This method extends the parent method by adding Fibonacci-based signals
        and implementing confluence-based entry rules.
        """
        # Check time validity from parent class
        hour = row.name.hour if isinstance(row.name, pd.Timestamp) else row.name.hour
        if hour < self.trading_hours['start'] or hour >= self.trading_hours['end']:
            return None
        
        # Get indicator-based signal from parent method
        indicator_signal = super().check_entry_signals(row)
        
        if not self.use_fibonacci:
            return indicator_signal
            
        # Check for Fibonacci signal
        fib_signal = row.get('fib_signal', None)
        fib_level = row.get('fib_level_touched', None)
        
        # Print debugging info
        if fib_signal is not None:
            print(f"Fibonacci signal: {fib_signal} at {fib_level:.3f} level")
        
        if indicator_signal is not None:
            print(f"Indicator signal: {indicator_signal}")
        
        # Both signals agree - strongest case (confluence)
        if indicator_signal == fib_signal and indicator_signal is not None:
            print(f"Strong confluence signal: Indicators and Fibonacci {fib_level:.3f} both suggest {indicator_signal}")
            
            # Track stats
            if fib_level is not None:
                level_key = str(round(fib_level, 3))
                if level_key in self.fib_stats:
                    self.fib_stats[level_key]['count'] += 1
            
            self.confluence_count += 1
            self.total_signals += 1
            
            # Strong confluence signal
            return indicator_signal
        
        # Only Fibonacci signal - use specific levels
        if fib_signal is not None and indicator_signal is None:
            if fib_level is not None:
                # Higher probability levels get priority
                if fib_level in [0.382, 0.5, 0.618]:
                    print(f"Using Fibonacci signal at key level {fib_level:.3f}")
                    
                    # Track stats
                    level_key = str(round(fib_level, 3))
                    if level_key in self.fib_stats:
                        self.fib_stats[level_key]['count'] += 1
                    
                    self.total_signals += 1
                    
                    return fib_signal
                else:
                    print(f"Weak Fibonacci level {fib_level:.3f} - waiting for confirmation")
        
        # Only indicator signal - use standard approach
        if indicator_signal is not None:
            self.total_signals += 1
            return indicator_signal
        
        # No signal
        return None
    
    def open_position(self, signal, row, market_regime="normal_volatility"):
        """
        Open position with Fibonacci-enhanced stop loss and take profit
        
        Extends the parent method by using Fibonacci-based levels for
        more precise stop loss and take profit placement.
        """
        # Check if we have Fibonacci-based stop loss and take profit
        fib_stop = None
        fib_tp = None
        
        if 'fib_stop_loss' in row and row['fib_stop_loss'] is not None:
            fib_stop = row['fib_stop_loss']
        
        if 'fib_take_profit1' in row and row['fib_take_profit1'] is not None:
            # Use the first take profit level (61.8% extension)
            fib_tp = row['fib_take_profit1']
        
        # If we prefer Fibonacci stops and have them, use them
        if self.prefer_fibonacci_stops and fib_stop is not None:
            # Store original parameters
            original_sl_points = self.stop_loss_points
            original_tp_points = self.take_profit_points
            
            fib_level = row.get('fib_level_touched', None)
            level_str = f"{fib_level:.3f}" if fib_level is not None else "Unknown"
            
            print(f"Using Fibonacci {level_str} level-based stop loss at {fib_stop:.3f}")
            
            # Calculate new stop loss points
            if signal == 'buy':
                sl_distance = abs(row['close'] - fib_stop)
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # If Fibonacci take profit available, use it
                if fib_tp is not None:
                    tp_distance = abs(fib_tp - row['close'])
                    tp_points = int(tp_distance * 1000)  # Convert to points
                    
                    # Temporarily set stop and target
                    self.stop_loss_points = sl_points
                    self.take_profit_points = tp_points
                    
                    print(f"Using Fibonacci-based target at {fib_tp:.3f}")
                    
                    # Call parent method with modified parameters
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    self.take_profit_points = original_tp_points
                    
                    return result
                else:
                    # Use Fibonacci stop with standard target
                    self.stop_loss_points = sl_points
                    
                    # Call parent method
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    
                    return result
            
            elif signal == 'sell':
                sl_distance = abs(fib_stop - row['close'])
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # If Fibonacci take profit available, use it
                if fib_tp is not None:
                    tp_distance = abs(row['close'] - fib_tp)
                    tp_points = int(tp_distance * 1000)  # Convert to points
                    
                    # Temporarily set stop and target
                    self.stop_loss_points = sl_points
                    self.take_profit_points = tp_points
                    
                    print(f"Using Fibonacci-based target at {fib_tp:.3f}")
                    
                    # Call parent method with modified parameters
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    self.take_profit_points = original_tp_points
                    
                    return result
                else:
                    # Use Fibonacci stop with standard target
                    self.stop_loss_points = sl_points
                    
                    # Call parent method
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    
                    return result
        
        # No Fibonacci levels, use parent implementation
        return super().open_position(signal, row, market_regime)
    
    def close_position(self, row, profit, reason):
        """
        Close position and update Fibonacci statistics
        
        Extends the parent method by tracking Fibonacci level performance
        """
        # Get Fibonacci level for the current trade if applicable
        fib_level = getattr(self, 'current_fib_level', None)
        
        # Call parent method to handle the actual position closing
        result = super().close_position(row, profit, reason)
        
        # Update Fibonacci statistics if applicable
        if fib_level is not None:
            level_key = str(round(fib_level, 3))
            if level_key in self.fib_stats:
                if profit > 0:
                    self.fib_stats[level_key]['wins'] += 1
                else:
                    self.fib_stats[level_key]['losses'] += 1
                
                # Calculate win rate for Fibonacci level
                total = self.fib_stats[level_key]['wins'] + self.fib_stats[level_key]['losses']
                if total > 0:
                    win_rate = (self.fib_stats[level_key]['wins'] / total) * 100
                    print(f"Fibonacci level {level_key} win rate: {win_rate:.1f}% ({self.fib_stats[level_key]['wins']}/{total})")
        
        # Reset current Fibonacci level
        self.current_fib_level = None
        
        return result
    
    def backtest(self, csv_file, *, use_timeframe=None):
        """
        Run backtest with Fibonacci analysis
        
        Extends the parent method to include Fibonacci statistics in results
        """
        # Run standard backtest
        results = super().backtest(csv_file, use_timeframe=use_timeframe)
        
        # Add Fibonacci statistics to results
        if self.use_fibonacci and results:
            results['fibonacci_stats'] = self.fib_stats
            
            # Calculate confluence performance
            if self.total_signals > 0:
                confluence_percentage = (self.confluence_count / self.total_signals) * 100
                results['confluence_percentage'] = confluence_percentage
                
                print(f"\n=== Fibonacci Analysis Performance ===")
                print(f"Total signals: {self.total_signals}")
                print(f"Confluence signals: {self.confluence_count} ({confluence_percentage:.1f}%)")
                
                # Print individual level performance
                print("\nPerformance by Fibonacci level:")
                for level, stats in self.fib_stats.items():
                    if stats['count'] > 0:
                        wins = stats['wins']
                        total_results = stats['wins'] + stats['losses']
                        win_rate = (wins / total_results) * 100 if total_results > 0 else 0
                        print(f"Level {level}: {win_rate:.1f}% win rate ({stats['count']} trades, {wins}/{total_results} completed)")
        
        return results
        
    def generate_trade_report(self, output_file=None):
        """
        Generate comprehensive trade report with Fibonacci analysis
        
        Extends the parent method to include Fibonacci statistics in report
        """
        # Get standard report from parent
        standard_report = super().generate_trade_report(output_file=None)
        
        # If no trades, create a basic report message
        if standard_report is None:
            standard_report = "No trades to report.\n\n"
        
        if not self.use_fibonacci:
            # If not using Fibonacci, just use standard report
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(standard_report)
                print(f"Report saved to {output_file}")
            return standard_report
        
        # Add Fibonacci analysis section
        fib_report = "\n\nFibonacci Retracement Analysis\n" + "-" * 50 + "\n"
        fib_report += f"Fibonacci analysis enabled: {self.use_fibonacci}\n"
        fib_report += f"Fibonacci lookback: {self.fibonacci_lookback}\n"
        fib_report += f"Swing point lookback: {self.swing_lookback}\n"
        fib_report += f"Trend window: {self.fib_trend_window}\n"
        fib_report += f"Prefer Fibonacci stops: {self.prefer_fibonacci_stops}\n\n"
        
        # Add confluence statistics
        if hasattr(self, 'total_signals') and self.total_signals > 0:
            confluence_percentage = (self.confluence_count / self.total_signals) * 100
            fib_report += f"Total signals: {self.total_signals}\n"
            fib_report += f"Confluence signals: {self.confluence_count} ({confluence_percentage:.1f}%)\n\n"
        
        # Add Fibonacci level statistics
        fib_report += "Fibonacci Level Performance:\n"
        for level, stats in self.fib_stats.items():
            if stats['count'] > 0:
                wins = stats['wins']
                total_results = stats['wins'] + stats['losses']
                win_rate = (wins / total_results) * 100 if total_results > 0 else 0
                fib_report += f"Level {level:8s}: {stats['count']:3d} trades, {win_rate:5.1f}% win rate ({wins}/{total_results} completed)\n"
        
        # Combine reports
        combined_report = standard_report + fib_report
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_report)
            print(f"Report saved to {output_file}")
        
        return combined_report


    def plot_fibonacci_analysis(self, lookback=50):
        """
        Generate Fibonacci analysis chart for recent signals
        
        Parameters:
        -----------
        lookback : int
            Number of bars to look back for analysis
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with Fibonacci levels and analysis
        """
        # Get the prepared data with Fibonacci signals
        if not hasattr(self, 'prepared_data') or self.prepared_data is None or len(self.prepared_data) == 0:
            print("No data available for Fibonacci analysis plot")
            return None
        
        # Check if we have any Fibonacci signals
        if 'fib_retracement' not in self.prepared_data.columns or not self.prepared_data['fib_retracement'].any():
            print("No Fibonacci retracements found in data")
            return None
        
        # Plot Fibonacci levels using the utility function
        fig = plot_fibonacci_levels(self.prepared_data, lookback)
        
        return fig


# Factory function to create Fibonacci-enhanced bot
def create_fibonacci_bot(**kwargs):
    """
    Create a Fibonacci-enhanced XAUUSD trading bot
    
    Parameters:
    -----------
    **kwargs : dict
        Bot parameters, passed to XAUUSDFibonacciBot constructor
        
    Returns:
    --------
    XAUUSDFibonacciBot
        Bot instance with Fibonacci retracement analysis
    """
    return XAUUSDFibonacciBot(**kwargs)