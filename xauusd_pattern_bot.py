"""
Extended XAUUSD Trading Bot with Candlestick Pattern Recognition

This module extends the base XAUUSDCentScalpingBot by adding
candlestick pattern recognition and technical analysis rules.
"""

from xauusd_bot import XAUUSDCentScalpingBot
import pandas as pd
import numpy as np
from utils.candlestick_patterns import identify_patterns, find_support_resistance, set_profit_targets
from pattern_integration import integrate_patterns_with_bot

class XAUUSDPatternBot(XAUUSDCentScalpingBot):
    """
    Extended trading bot that combines the original MA/Stochastic strategy
    with candlestick pattern recognition techniques.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the pattern-enhanced trading bot
        
        Parameters:
        -----------
        **kwargs : dict
            Bot parameters, passed to parent class
            
        Additional parameters:
        ---------------------
        min_body_pct : float
            Minimum candle body size as percentage of price (default: 1.0)
        max_body_pct : float
            Maximum candle body size as percentage of price (default: 10.0)
        """
        # Extract pattern-specific parameters with defaults
        self.min_body_pct = kwargs.pop('min_body_pct', 1.0)
        self.max_body_pct = kwargs.pop('max_body_pct', 10.0)
        self.use_pattern_recognition = kwargs.pop('use_pattern_recognition', True)
        self.pattern_position_sizing = kwargs.pop('pattern_position_sizing', True)
        self.risk_averse_entry = kwargs.pop('risk_averse_entry', True)
        
        # Initialize parent class with remaining parameters
        super().__init__(**kwargs)
        
        # Add pattern-specific attributes
        self.support_resistance_levels = []
        self.pattern_signals = pd.DataFrame()
        
        # Tracking for candlestick pattern stats
        self.pattern_stats = {
            'marubozu_bullish': {'count': 0, 'wins': 0, 'losses': 0},
            'marubozu_bearish': {'count': 0, 'wins': 0, 'losses': 0},
            'hammer': {'count': 0, 'wins': 0, 'losses': 0},
            'hanging_man': {'count': 0, 'wins': 0, 'losses': 0},
            'shooting_star': {'count': 0, 'wins': 0, 'losses': 0},
            'engulfing_bullish': {'count': 0, 'wins': 0, 'losses': 0},
            'engulfing_bearish': {'count': 0, 'wins': 0, 'losses': 0},
            'harami_bullish': {'count': 0, 'wins': 0, 'losses': 0},
            'harami_bearish': {'count': 0, 'wins': 0, 'losses': 0},
            'piercing': {'count': 0, 'wins': 0, 'losses': 0},
            'dark_cloud': {'count': 0, 'wins': 0, 'losses': 0},
            'morning_star': {'count': 0, 'wins': 0, 'losses': 0},
            'evening_star': {'count': 0, 'wins': 0, 'losses': 0}
        }
        
        # Initialize last_pattern attribute
        self.last_pattern = None
    def prepare_data(self, df):
        """
        Prepare data with indicators and pattern recognition
        """
        # Use parent method first to get technical indicators
        result = super().prepare_data(df)
        
        if self.use_pattern_recognition:
            print(f"\nPattern Recognition Debugging:")
            print(f"Input data shape: {result.shape}")
            
            # Add pattern recognition
            pattern_df = identify_patterns(
                result, 
                min_body_pct=self.min_body_pct, 
                max_body_pct=self.max_body_pct
            )
            
            # Count pattern occurrences
            pattern_counts = {}
            for pattern in ['marubozu_bullish', 'marubozu_bearish', 'hammer', 
                        'hanging_man', 'shooting_star', 'engulfing_bullish',
                        'engulfing_bearish', 'harami_bullish', 'harami_bearish',
                        'piercing', 'dark_cloud', 'morning_star', 'evening_star']:
                pattern_counts[pattern] = pattern_df[pattern].sum()
            
            print("\nPattern counts:")
            for pattern, count in pattern_counts.items():
                print(f"{pattern}: {count}")
                
            # Count valid candles for trading
            valid_candles = pattern_df['valid_trade'].sum()
            print(f"\nValid candles for trading: {valid_candles} out of {len(pattern_df)}")
            
            # Count signal types
            signal_counts = pattern_df['pattern_signal'].value_counts()
            print("\nSignal counts:")
            print(signal_counts)
            
            # Find support and resistance levels
            self.support_resistance_levels = find_support_resistance(pattern_df, lookback=100)
            print(f"\nSupport/Resistance levels found: {len(self.support_resistance_levels)}")
            
       
            
            # Store pattern signals for reference
            self.pattern_signals = pattern_df[
                [
                    'marubozu_bullish', 'marubozu_bearish', 
                    'hammer', 'hanging_man', 'shooting_star',
                    'engulfing_bullish', 'engulfing_bearish',
                    'harami_bullish', 'harami_bearish',
                    'piercing', 'dark_cloud',
                    'morning_star', 'evening_star',
                    'pattern_signal', 'pattern_stop_loss', 'pattern_target',
                    'pattern_name'
                ]
            ]
            
            # Merge pattern signals into result
            for col in self.pattern_signals.columns:
                result[col] = self.pattern_signals[col]
        
        return result
    
    def check_entry_signals(self, row):
        """
        Check for entry signals with both indicators and patterns
        
        This method extends the parent method by adding pattern-based signals
        and implementing risk-taker vs risk-averse entry rules.
        """
        # Skip candle body size check temporarily for debugging
        # Comment out the body percentage check to be more lenient
        # body_pct = (abs(row['close'] - row['open']) / row['close']) * 100
        # if body_pct < self.min_body_pct or body_pct > self.max_body_pct:
        #     print(f"Candle body size {body_pct:.2f}% outside valid range ({self.min_body_pct}%-{self.max_body_pct}%)")
        #     return None
            
        # Check if time is valid for trading from parent class
        hour = row.name.hour if isinstance(row.name, pd.Timestamp) else row.name.hour
        if hour < self.trading_hours['start'] or hour >= self.trading_hours['end']:
            return None
        
        # Get indicator-based signal from parent method
        indicator_signal = super().check_entry_signals(row)
        
        if not self.use_pattern_recognition:
            return indicator_signal
            
        # Check for pattern signal
        pattern_signal = row.get('pattern_signal', None)
        pattern_name = row.get('pattern_name', None)
        
        # Print debugging info
        if pattern_signal is not None:
            print(f"Pattern signal: {pattern_signal} from {pattern_name}")
        
        if indicator_signal is not None:
            print(f"Indicator signal: {indicator_signal}")
        
        # Skip risk-averse mode check to be more aggressive
        # if self.risk_averse_entry and pattern_signal is not None:
        #     if (pattern_signal == 'buy' and not row['is_bullish']) or (pattern_signal == 'sell' and not row['is_bearish']):
        #         print(f"Risk-averse mode: Waiting for confirmation on {pattern_name}")
        #         return None
        
        # Both signals agree - strongest case
        if indicator_signal == pattern_signal and indicator_signal is not None:
            if pattern_name:
                print(f"Strong signal: Indicators and {pattern_name} pattern both suggest {indicator_signal}")
                
                # Update pattern stats
                if pattern_name in self.pattern_stats:
                    self.pattern_stats[pattern_name]['count'] += 1
                    
            return indicator_signal
        
        # MODIFICATION: Always use either indicator or pattern signal, whichever is available
        if pattern_signal is not None:
            # Update pattern stats
            if pattern_name in self.pattern_stats:
                self.pattern_stats[pattern_name]['count'] += 1
            
            print(f"Using pattern signal: {pattern_signal} from {pattern_name}")
            return pattern_signal
        
        # Default to indicator signal
        return indicator_signal
    
    def open_position(self, signal, row, market_regime="normal_volatility"):
        """
        Open position with pattern-enhanced stop loss and take profit
        
        Extends the parent method by using pattern-based stop levels and targets
        when available.
        """
        # Check if we have pattern-based stop loss
        if 'pattern_stop_loss' in row and row['pattern_stop_loss'] is not None:
            # Store original parameters
            original_sl_points = self.stop_loss_points
            original_tp_points = self.take_profit_points
            
            pattern_name = row.get('pattern_name', 'Unknown pattern')
            
            print(f"Using {pattern_name} stop loss at {row['pattern_stop_loss']:.3f}")
            
            # Calculate new stop loss points
            if signal == 'buy':
                sl_distance = abs(row['close'] - row['pattern_stop_loss'])
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # If pattern has target, use that instead of standard TP
                if 'pattern_target' in row and row['pattern_target'] is not None:
                    tp_distance = abs(row['pattern_target'] - row['close'])
                    tp_points = int(tp_distance * 1000)  # Convert to points
                    
                    # Temporarily set stop and target
                    self.stop_loss_points = sl_points
                    self.take_profit_points = tp_points
                    
                    print(f"Using pattern-based target at {row['pattern_target']:.3f}")
                    
                    # Call parent method with modified parameters
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    self.take_profit_points = original_tp_points
                    
                    return result
                else:
                    # Use pattern stop with standard target
                    self.stop_loss_points = sl_points
                    
                    # Call parent method
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    
                    return result
            
            elif signal == 'sell':
                sl_distance = abs(row['pattern_stop_loss'] - row['close'])
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # If pattern has target, use that instead of standard TP
                if 'pattern_target' in row and row['pattern_target'] is not None:
                    tp_distance = abs(row['close'] - row['pattern_target'])
                    tp_points = int(tp_distance * 1000)  # Convert to points
                    
                    # Temporarily set stop and target
                    self.stop_loss_points = sl_points
                    self.take_profit_points = tp_points
                    
                    print(f"Using pattern-based target at {row['pattern_target']:.3f}")
                    
                    # Call parent method with modified parameters
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    self.take_profit_points = original_tp_points
                    
                    return result
                else:
                    # Use pattern stop with standard target
                    self.stop_loss_points = sl_points
                    
                    # Call parent method
                    result = super().open_position(signal, row, market_regime)
                    
                    # Restore original parameters
                    self.stop_loss_points = original_sl_points
                    
                    return result
        
        # No pattern-based stop, use parent implementation
        return super().open_position(signal, row, market_regime)
    
    def calculate_lot_size(self, stop_loss_points):
        """
        Calculate position size with pattern-based adjustments
        
        Extends the parent method by adjusting position size based on
        pattern confidence when enabled.
        """
        # Get base position size from parent
        base_lot_size = super().calculate_lot_size(stop_loss_points)
        
        if not self.pattern_position_sizing:
            return base_lot_size
            
        # Adjust based on most recent pattern
        if hasattr(self, 'last_pattern'):
            pattern_name = self.last_pattern
            
            # Stronger patterns get larger position sizes
            if pattern_name in ['Morning Star', 'Evening Star']:
                adjusted_size = base_lot_size * 1.0  # 100% of base size for strongest patterns
                print(f"Strong pattern ({pattern_name}): Using 100% position size")
                return adjusted_size
            elif pattern_name in ['Bullish Engulfing', 'Bearish Engulfing']:
                adjusted_size = base_lot_size * 0.8  # 80% for strong patterns
                print(f"Strong pattern ({pattern_name}): Using 80% position size")
                return adjusted_size
            elif pattern_name in ['Piercing', 'Dark Cloud', 'Harami Bullish', 'Harami Bearish']:
                adjusted_size = base_lot_size * 0.6  # 60% for moderate patterns
                print(f"Moderate pattern ({pattern_name}): Using 60% position size")
                return adjusted_size
            else:
                adjusted_size = base_lot_size * 0.4  # 40% for weaker patterns
                print(f"Weaker pattern ({pattern_name}): Using 40% position size")
                return adjusted_size
                
        # No pattern, use standard sizing
        return base_lot_size
    
    def close_position(self, row, profit, reason):
        """
        Close position and update pattern statistics
        
        Extends the parent method by tracking pattern success rates
        """
        # Get info about the trade for statistics
        current_pattern = getattr(self, 'last_pattern', None)
        
        # Call parent method to handle the actual position closing
        result = super().close_position(row, profit, reason)
        
        # Update pattern statistics if applicable
        if current_pattern and current_pattern in self.pattern_stats:
            if profit > 0:
                self.pattern_stats[current_pattern]['wins'] += 1
            else:
                self.pattern_stats[current_pattern]['losses'] += 1
                
            # Calculate win rate for pattern
            total = self.pattern_stats[current_pattern]['wins'] + self.pattern_stats[current_pattern]['losses']
            if total > 0:
                win_rate = (self.pattern_stats[current_pattern]['wins'] / total) * 100
                print(f"Pattern {current_pattern} win rate: {win_rate:.1f}% ({self.pattern_stats[current_pattern]['wins']}/{total})")
        
        # Reset current pattern
        self.last_pattern = None
        
        return result
    
    def backtest(self, csv_file, *, use_timeframe=None):
        """
        Run backtest with pattern recognition
        
        Extends the parent method to include pattern statistics in results
        """
        # Run standard backtest
        results = super().backtest(csv_file, use_timeframe=use_timeframe)
        
        # Add pattern statistics to results
        if self.use_pattern_recognition and results:
            results['pattern_stats'] = self.pattern_stats
            
            # Calculate overall pattern performance
            total_pattern_trades = sum(p['count'] for p in self.pattern_stats.values())
            total_pattern_wins = sum(p['wins'] for p in self.pattern_stats.values())
            
            if total_pattern_trades > 0:
                pattern_win_rate = (total_pattern_wins / total_pattern_trades) * 100
                results['pattern_win_rate'] = pattern_win_rate
                
                print(f"\n=== Pattern Recognition Performance ===")
                print(f"Total trades with patterns: {total_pattern_trades}")
                print(f"Pattern win rate: {pattern_win_rate:.1f}%")
                
                # Print individual pattern performance
                print("\nPerformance by pattern:")
                for pattern, stats in self.pattern_stats.items():
                    if stats['count'] > 0:
                        win_rate = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
                        print(f"{pattern}: {win_rate:.1f}% win rate ({stats['wins']}/{stats['count']} trades)")
        
        return results
    
    def generate_trade_report(self, output_file=None):
        """
        Generate comprehensive trade report with pattern analysis
        
        Extends the parent method to include pattern statistics in report
        """
        # Get standard report from parent
        standard_report = super().generate_trade_report(output_file=None)
        
        # If no trades, create a basic report message
        if standard_report is None:
            standard_report = "No trades to report.\n\n"
        
        if not self.use_pattern_recognition:
            # If not using patterns, just use standard report
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(standard_report)
                print(f"Report saved to {output_file}")
            return standard_report
        
        # Add pattern analysis section
        pattern_report = "\n\nPattern Recognition Analysis\n" + "-" * 50 + "\n"
        pattern_report += f"Pattern recognition enabled: {self.use_pattern_recognition}\n"
        pattern_report += f"Pattern position sizing: {self.pattern_position_sizing}\n"
        pattern_report += f"Risk-averse entry mode: {self.risk_averse_entry}\n\n"
        
        # Add pattern statistics
        pattern_report += "Pattern Performance:\n"
        for pattern, stats in self.pattern_stats.items():
            if stats['count'] > 0:
                win_rate = (stats['wins'] / (stats['wins'] + stats['losses'])) * 100 if stats['wins'] + stats['losses'] > 0 else 0
                pattern_report += f"{pattern:20s}: {stats['count']} trades, {win_rate:.1f}% win rate ({stats['wins']}/{stats['wins'] + stats['losses']})\n"
        
        # Combine reports
        combined_report = standard_report + pattern_report
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_report)
            print(f"Report saved to {output_file}")
        
        return combined_report


# Easy factory function to create pattern-enhanced bot
def create_pattern_bot(**kwargs):
    """
    Create a pattern-enhanced XAUUSD trading bot
    
    Parameters:
    -----------
    **kwargs : dict
        Bot parameters, passed to XAUUSDPatternBot constructor
        
    Returns:
    --------
    XAUUSDPatternBot
        Bot instance with pattern recognition
    """
    return XAUUSDPatternBot(**kwargs)