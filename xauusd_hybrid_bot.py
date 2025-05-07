"""
Hybrid XAUUSD Trading Bot

This bot combines the strengths of the standard XAUUSDCentScalpingBot with
Fibonacci retracement analysis for improved entry and exit precision.
"""

from xauusd_bot import XAUUSDCentScalpingBot
import pandas as pd
import numpy as np
from utils.fibonacci import generate_fibonacci_signals, plot_fibonacci_levels

class XAUUSDHybridBot(XAUUSDCentScalpingBot):
    """
    Hybrid trading bot that uses the standard bot's signals but enhances
    them with Fibonacci analysis for improved entry/exit precision.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the hybrid trading bot
        
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
        prioritize_level_382 : bool
            Whether to prioritize the 0.382 Fibonacci level (default: True)
        fibonacci_filter_only : bool
            Use Fibonacci only for filtering, not for signal generation (default: True)
        enhanced_exits : bool
            Use Fibonacci for improved exit points (default: True)
        """
        # Extract hybrid-specific parameters with defaults
        self.use_fibonacci = kwargs.pop('use_fibonacci', True)
        self.fibonacci_lookback = kwargs.pop('fibonacci_lookback', 50)
        self.swing_lookback = kwargs.pop('swing_lookback', 10)
        self.fib_trend_window = kwargs.pop('fib_trend_window', 20)
        self.prioritize_level_382 = kwargs.pop('prioritize_level_382', True)
        self.fibonacci_filter_only = kwargs.pop('fibonacci_filter_only', True)
        self.enhanced_exits = kwargs.pop('enhanced_exits', True)
        
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
        
        # Record enhanced stats
        self.standard_signals = 0
        self.enhanced_entries = 0
        self.enhanced_exits = 0
        self.filtered_signals = 0  # Signals filtered out by Fibonacci
        
    def prepare_data(self, df):
        """
        Prepare data with indicators and Fibonacci analysis
        """
        # Use parent method first to get technical indicators
        result = super().prepare_data(df)
        
        if self.use_fibonacci:
            # Generate Fibonacci signals
            fib_df = generate_fibonacci_signals(
                result,
                lookback=self.fibonacci_lookback,
                swing_lookback=self.swing_lookback,
                trend_window=self.fib_trend_window
            )
            
            # Store Fibonacci data
            self.fibonacci_signals = fib_df[
                [
                    'fib_retracement', 'fib_level_touched',
                    'fib_start_price', 'fib_end_price',
                    'fib_signal', 'fib_stop_loss',
                    'fib_take_profit1', 'fib_take_profit2', 'fib_take_profit3',
                    'trend'
                ]
            ] if set(['fib_retracement', 'fib_level_touched', 'fib_signal', 'trend']).issubset(fib_df.columns) else pd.DataFrame()
            
            # Merge Fibonacci columns into result
            for col in self.fibonacci_signals.columns:
                if col in self.fibonacci_signals:
                    result[col] = self.fibonacci_signals[col]
        
        return result
    
    def check_entry_signals(self, row):
        """
        Check for entry signals with Fibonacci confirmation/filtering
        
        This method enhances the parent method by using Fibonacci levels
        to confirm or filter standard signals.
        """
        # First check time validity from parent class
        hour = row.name.hour if isinstance(row.name, pd.Timestamp) else row.name.hour
        if hour < self.trading_hours['start'] or hour >= self.trading_hours['end']:
            return None
        
        # Get indicator-based signal from parent method
        # This is the core of your standard strategy
        indicator_signal = super().check_entry_signals(row)
        
        # If Fibonacci analysis is disabled, use standard signals
        if not self.use_fibonacci:
            if indicator_signal:
                self.standard_signals += 1
            return indicator_signal
        
        # Check for Fibonacci data in the row
        has_fib_data = all(col in row.index for col in ['fib_retracement', 'fib_level_touched', 'fib_signal'])
        
        # If no Fibonacci data, use standard signal
        if not has_fib_data:
            if indicator_signal:
                self.standard_signals += 1
            return indicator_signal
        
        # Get Fibonacci signal and level
        fib_signal = row.get('fib_signal', None)
        fib_level = row.get('fib_level_touched', None)
        fib_retracement = row.get('fib_retracement', False)
        
        # If prioritizing 0.382 level and signal doesn't match criteria, use standard
        if self.prioritize_level_382 and fib_level is not None and fib_level != 0.382 and not (0.37 <= fib_level <= 0.39):
            if indicator_signal:
                self.standard_signals += 1
            return indicator_signal
        
        # Both signals agree - highly reliable confluence signal
        if indicator_signal == fib_signal and indicator_signal is not None:
            # Track stats for this Fibonacci level
            if fib_level is not None:
                level_key = str(round(fib_level, 3))
                if level_key in self.fib_stats:
                    self.fib_stats[level_key]['count'] += 1
            
            # Store the Fibonacci level for position management
            self.current_fib_level = fib_level
            self.enhanced_entries += 1
            
            return indicator_signal
        
        # Only indicator signal - filter with Fibonacci context
        if indicator_signal is not None and fib_signal is None:
            # Use standard signal but store context
            self.standard_signals += 1
            
            # Check if the signal aligns with the detected trend
            trend = row.get('trend', 'neutral')
            if (indicator_signal == 'buy' and trend == 'uptrend') or (indicator_signal == 'sell' and trend == 'downtrend'):
                # Signal matches trend - good
                return indicator_signal
            elif self.fibonacci_filter_only and ((indicator_signal == 'buy' and trend == 'downtrend') or (indicator_signal == 'sell' and trend == 'uptrend')):
                # Signal contradicts trend - filter out
                self.filtered_signals += 1
                return None
            else:
                # Don't filter if not set to filter only
                return indicator_signal
                
        # Only use Fibonacci for filtering, not signal generation
        if self.fibonacci_filter_only:
            return indicator_signal
                
        # Default to indicator signal
        if indicator_signal:
            self.standard_signals += 1
        return indicator_signal
    
    def open_position(self, signal, row, market_regime="normal_volatility"):
        """
        Open position with enhanced stop loss and take profit
        
        Uses Fibonacci levels for more precise position management
        when available.
        """
        # Store the current Fibonacci level and signal for position management
        self.entry_fib_level = row.get('fib_level_touched', None)
        self.entry_fib_signal = row.get('fib_signal', None)
        
        # Check if we have Fibonacci-based stop loss and take profit
        has_fib_exits = all(col in row.index for col in ['fib_stop_loss', 'fib_take_profit1'])
        fib_stop = row.get('fib_stop_loss', None) if has_fib_exits else None
        fib_tp = row.get('fib_take_profit1', None) if has_fib_exits else None
        
        # If using enhanced exits and have Fibonacci levels, use them
        if self.enhanced_exits and fib_stop is not None and signal == self.entry_fib_signal:
            # Store original parameters
            original_sl_points = self.stop_loss_points
            original_tp_points = self.take_profit_points
            
            # Calculate new stop loss and target points based on Fibonacci
            if signal == 'buy':
                sl_distance = abs(row['close'] - fib_stop)
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # Use Fibonacci take profit if available
                if fib_tp is not None:
                    tp_distance = abs(fib_tp - row['close'])
                    tp_points = int(tp_distance * 1000)  # Convert to points
                    
                    # Only use fib TP if sufficiently large and respects min R:R
                    r_r_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                    if r_r_ratio >= self.r_r_ratio and tp_points >= 20:
                        # Temporarily set stop and target
                        self.stop_loss_points = sl_points
                        self.take_profit_points = tp_points
                        self.enhanced_exits += 1
                        
                        # Call parent method with modified parameters
                        result = super().open_position(signal, row, market_regime)
                        
                        # Restore original parameters
                        self.stop_loss_points = original_sl_points
                        self.take_profit_points = original_tp_points
                        
                        return result
            
            elif signal == 'sell':
                sl_distance = abs(fib_stop - row['close'])
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # Use Fibonacci take profit if available
                if fib_tp is not None:
                    tp_distance = abs(row['close'] - fib_tp)
                    tp_points = int(tp_distance * 1000)  # Convert to points
                    
                    # Only use fib TP if sufficiently large and respects min R:R
                    r_r_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                    if r_r_ratio >= self.r_r_ratio and tp_points >= 20:
                        # Temporarily set stop and target
                        self.stop_loss_points = sl_points
                        self.take_profit_points = tp_points
                        self.enhanced_exits += 1
                        
                        # Call parent method with modified parameters
                        result = super().open_position(signal, row, market_regime)
                        
                        # Restore original parameters
                        self.stop_loss_points = original_sl_points
                        self.take_profit_points = original_tp_points
                        
                        return result
        
        # Use standard approach if no Fibonacci enhancements applied
        return super().open_position(signal, row, market_regime)
    
    def close_position(self, row, profit, reason):
        """
        Close position and update Fibonacci statistics
        """
        # Get Fibonacci level for the current trade
        fib_level = getattr(self, 'entry_fib_level', None)
        
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
        
        # Reset current trade info
        self.entry_fib_level = None
        self.entry_fib_signal = None
        
        return result
    
    def backtest(self, csv_file, *, use_timeframe=None):
        """
        Run backtest with hybrid approach
        """
        # Reset stats before backtest
        self.standard_signals = 0
        self.enhanced_entries = 0
        self.enhanced_exits = 0
        self.filtered_signals = 0
        
        # Run standard backtest
        results = super().backtest(csv_file, use_timeframe=use_timeframe)
        
        # Add hybrid statistics to results
        if self.use_fibonacci and results:
            # Add Fibonacci statistics
            results['fibonacci_stats'] = self.fib_stats
            
            # Add enhanced entry stats
            results['standard_signals'] = self.standard_signals
            results['enhanced_entries'] = self.enhanced_entries
            results['enhanced_exits'] = self.enhanced_exits
            results['filtered_signals'] = self.filtered_signals
            
            # Add enhancement percentages
            if self.standard_signals > 0:
                results['enhanced_entry_pct'] = (self.enhanced_entries / self.standard_signals) * 100
            else:
                results['enhanced_entry_pct'] = 0
                
            if len(self.trades) > 0:
                results['enhanced_exit_pct'] = (self.enhanced_exits / len(self.trades)) * 100
            else:
                results['enhanced_exit_pct'] = 0
            
            # Print summary
            print(f"\n=== Hybrid Strategy Performance ===")
            print(f"Standard signals: {self.standard_signals}")
            print(f"Enhanced entries: {self.enhanced_entries} ({results['enhanced_entry_pct']:.1f}%)")
            print(f"Enhanced exits: {self.enhanced_exits} ({results['enhanced_exit_pct']:.1f}%)")
            print(f"Filtered signals: {self.filtered_signals}")
            
            # Print Fibonacci level performance
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
        Generate comprehensive trade report with hybrid analysis
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
        
        # Add hybrid analysis section
        hybrid_report = "\n\nHybrid Strategy Analysis\n" + "-" * 50 + "\n"
        hybrid_report += f"Hybrid strategy parameters:\n"
        hybrid_report += f"- Fibonacci analysis: {self.use_fibonacci}\n"
        hybrid_report += f"- Fibonacci lookback: {self.fibonacci_lookback}\n"
        hybrid_report += f"- Swing point lookback: {self.swing_lookback}\n"
        hybrid_report += f"- Fibonacci trend window: {self.fib_trend_window}\n"
        hybrid_report += f"- Prioritize 0.382 level: {self.prioritize_level_382}\n"
        hybrid_report += f"- Fibonacci filter only: {self.fibonacci_filter_only}\n"
        hybrid_report += f"- Enhanced exits: {self.enhanced_exits}\n\n"
        
        # Add strategy stats
        if hasattr(self, 'standard_signals'):
            hybrid_report += f"Standard signals: {self.standard_signals}\n"
            hybrid_report += f"Enhanced entries: {self.enhanced_entries}"
            
            if self.standard_signals > 0:
                enhanced_entry_pct = (self.enhanced_entries / self.standard_signals) * 100
                hybrid_report += f" ({enhanced_entry_pct:.1f}%)"
            
            hybrid_report += "\n"
            hybrid_report += f"Enhanced exits: {self.enhanced_exits}"
            
            if len(self.trades) > 0:
                enhanced_exit_pct = (self.enhanced_exits / len(self.trades)) * 100
                hybrid_report += f" ({enhanced_exit_pct:.1f}%)"
                
            hybrid_report += "\n"
            hybrid_report += f"Filtered signals: {self.filtered_signals}\n\n"
        
        # Add Fibonacci level statistics
        hybrid_report += "Fibonacci Level Performance:\n"
        for level, stats in self.fib_stats.items():
            if stats['count'] > 0:
                wins = stats['wins']
                total_results = stats['wins'] + stats['losses']
                win_rate = (wins / total_results) * 100 if total_results > 0 else 0
                hybrid_report += f"Level {level:8s}: {stats['count']:3d} trades, {win_rate:5.1f}% win rate ({wins}/{total_results} completed)\n"
        
        # Combine reports
        combined_report = standard_report + hybrid_report
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_report)
            print(f"Report saved to {output_file}")
        
        return combined_report
    
    def plot_fibonacci_analysis(self, lookback=50):
        """
        Generate Fibonacci analysis chart for recent signals
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


# Factory function to create hybrid bot
def create_hybrid_bot(**kwargs):
    """
    Create a hybrid XAUUSD trading bot
    
    Parameters:
    -----------
    **kwargs : dict
        Bot parameters
        
    Returns:
    --------
    XAUUSDHybridBot
        Bot instance with hybrid approach
    """
    return XAUUSDHybridBot(**kwargs)