"""
Enhanced Hybrid XAUUSD Trading Bot

This bot combines the strengths of the standard XAUUSDCentScalpingBot with
Fibonacci retracement analysis for improved entry and exit precision.
Includes advanced parameters for optimized performance.
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
            Bars to look back for Fibonacci analysis (default: 35)
        swing_lookback : int
            Bars to look for swing points (default: 8)
        fib_trend_window : int
            Window for trend detection (default: 15)
        prioritize_level_382 : bool
            Whether to prioritize the 0.382 Fibonacci level (default: True)
        prioritize_level_618 : bool
            Whether to prioritize the 0.618 Fibonacci level (default: True)
        fibonacci_filter_only : bool
            Use Fibonacci only for filtering, not for signal generation (default: True)
        trend_confirmation : bool
            Only take trades in the direction of the overall trend (default: True)
        enhanced_exits : bool
            Use Fibonacci for improved exit points (default: True)
        use_multiple_targets : bool
            Use multiple take profit targets based on extensions (default: True)
        trailing_exit_factor : float
            Dynamic trailing stop based on ATR multiplier (default: 1.5)
        """
        # Extract hybrid-specific parameters with defaults
        self.use_fibonacci = kwargs.pop('use_fibonacci', True)
        self.fibonacci_lookback = kwargs.pop('fibonacci_lookback', 35)
        self.swing_lookback = kwargs.pop('swing_lookback', 8)
        self.fib_trend_window = kwargs.pop('fib_trend_window', 15)
        self.prioritize_level_382 = kwargs.pop('prioritize_level_382', True)
        self.prioritize_level_618 = kwargs.pop('prioritize_level_618', True)
        self.fibonacci_filter_only = kwargs.pop('fibonacci_filter_only', True)
        self.trend_confirmation = kwargs.pop('trend_confirmation', True)
        self.enhanced_exits = kwargs.pop('enhanced_exits', True)
        self.use_multiple_targets = kwargs.pop('use_multiple_targets', True)
        self.trailing_exit_factor = kwargs.pop('trailing_exit_factor', 1.5)
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Add Fibonacci-specific attributes
        self.fibonacci_levels = {}
        self.fibonacci_signals = pd.DataFrame()
        self.overall_trend = 'neutral'
        
        # MultiTarget TP handling
        self.partial_exits_count = 0
        self.partial_exits_profit = 0
        self.trailing_stops_activated = 0
        # Position management attributes
        self.tp_targets = []
        self.using_multiple_targets = False
        self.trailing_trigger_price = None
        self.trailing_stop_activated = False
        self.partial_exit_done = False
  

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
        self.trend_filtered_signals = 0  # Signals filtered out by trend confirmation
        
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
            columns_to_include = [
                'fib_retracement', 'fib_level_touched',
                'fib_start_price', 'fib_end_price',
                'fib_signal', 'fib_stop_loss',
                'fib_take_profit1', 'fib_take_profit2', 'fib_take_profit3',
                'trend'
            ]
            
            self.fibonacci_signals = fib_df[
                [col for col in columns_to_include if col in fib_df.columns]
            ]
            
            # Merge Fibonacci columns into result
            for col in self.fibonacci_signals.columns:
                if col in self.fibonacci_signals:
                    result[col] = self.fibonacci_signals[col]
                    
            # Determine overall market trend
            if 'trend' in self.fibonacci_signals.columns:
                # Get the last 20 trend values
                recent_trends = self.fibonacci_signals['trend'].iloc[-20:].value_counts()
                
                # Set overall trend based on majority
                if 'uptrend' in recent_trends and recent_trends['uptrend'] > recent_trends.get('downtrend', 0):
                    self.overall_trend = 'uptrend'
                elif 'downtrend' in recent_trends and recent_trends['downtrend'] > recent_trends.get('uptrend', 0):
                    self.overall_trend = 'downtrend'
                else:
                    self.overall_trend = 'neutral'
                    
                print(f"Overall market trend detected: {self.overall_trend}")
        
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
        
        # Check for trend confirmation if enabled
        if self.trend_confirmation and indicator_signal is not None:
            current_trend = row.get('trend', 'neutral')
            # If overall trend doesn't match signal, filter it out
            if (indicator_signal == 'buy' and self.overall_trend == 'downtrend') or \
               (indicator_signal == 'sell' and self.overall_trend == 'uptrend'):
                print(f"Signal {indicator_signal} filtered due to opposing {self.overall_trend}")
                self.trend_filtered_signals += 1
                return None
        
        # Prioritize specific Fibonacci levels if enabled
        use_fib_level = False
        if fib_level is not None:
            if self.prioritize_level_382 and (abs(fib_level - 0.382) < 0.01):  # Within 0.01 of 0.382
                use_fib_level = True
                print(f"Using priority level 0.382 (actual: {fib_level:.3f})")
            elif self.prioritize_level_618 and (abs(fib_level - 0.618) < 0.01):  # Within 0.01 of 0.618
                use_fib_level = True
                print(f"Using priority level 0.618 (actual: {fib_level:.3f})")
        
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
            
            print(f"Confluence signal: indicator and Fibonacci both suggest {indicator_signal}")
            return indicator_signal
        
        # Only indicator signal - filter with Fibonacci context
        if indicator_signal is not None and (fib_signal is None or fib_signal != indicator_signal):
            # Use standard signal but check context
            self.standard_signals += 1
            
            # Check if the signal aligns with the detected trend
            trend = row.get('trend', 'neutral')
            if (indicator_signal == 'buy' and trend == 'uptrend') or (indicator_signal == 'sell' and trend == 'downtrend'):
                # Signal matches trend - good
                return indicator_signal
            elif self.fibonacci_filter_only and ((indicator_signal == 'buy' and trend == 'downtrend') or 
                                               (indicator_signal == 'sell' and trend == 'uptrend')):
                # Signal contradicts trend - filter out
                self.filtered_signals += 1
                print(f"Filtered {indicator_signal} signal that contradicts {trend}")
                return None
            else:
                # Don't filter if not set to filter only
                return indicator_signal
        
        # Fibonacci level is priority and signal exists
        if use_fib_level and fib_signal is not None and not self.fibonacci_filter_only:
            # Use Fibonacci signal for priority levels
            print(f"Using Fibonacci {fib_level:.3f} level signal: {fib_signal}")
            return fib_signal
                
        # Default to indicator signal
        if indicator_signal:
            self.standard_signals += 1
        return indicator_signal
    
    def calculate_dynamic_stops(self, row, signal, entry_price):
        """
        Calculate dynamic stop loss and take profit levels based on ATR
        
        Parameters:
        -----------
        row : pandas.Series
            Current price bar
        signal : str
            Trade direction ('buy' or 'sell')
        entry_price : float
            Entry price
            
        Returns:
        --------
        tuple
            (stop_loss, take_profit, trailing_activation)
        """
        # Use ATR if available
        if 'atr' in row:
            atr = row['atr']
            stop_distance = atr * self.trailing_exit_factor
            
            # Calculate stop loss
            if signal == 'buy':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * self.r_r_ratio)
                trailing_activation = entry_price + (stop_distance * 0.5)  # Activate at 50% of TP
            else:  # sell
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * self.r_r_ratio)
                trailing_activation = entry_price - (stop_distance * 0.5)  # Activate at 50% of TP
                
            return stop_loss, take_profit, trailing_activation
        else:
            # Fallback to standard calculation
            return None, None, None
    
    def open_position(self, signal, row, market_regime="normal_volatility"):
        """
        Open position with enhanced stop loss and take profit
        
        Uses Fibonacci levels for more precise position management
        when available.
        """

        # Initialize trailing stop related attributes
        self.trailing_stop_activated = False
        self.partial_exit_done = False
    
        # Store the current Fibonacci level and signal for position management
        self.entry_fib_level = row.get('fib_level_touched', None)
        self.entry_fib_signal = row.get('fib_signal', None)
        
        # Check if we have Fibonacci-based stop loss and take profit
        has_fib_exits = all(col in row.index for col in ['fib_stop_loss', 'fib_take_profit1'])
        fib_stop = row.get('fib_stop_loss', None) if has_fib_exits else None
        
        # Get Fibonacci take profit targets if available
        fib_tp1 = row.get('fib_take_profit1', None) if has_fib_exits else None
        fib_tp2 = row.get('fib_take_profit2', None) if has_fib_exits else None
        fib_tp3 = row.get('fib_take_profit3', None) if has_fib_exits else None
        
        # Initialize multiple targets - these will be used for partial exits
        self.tp_targets = []
        
        # Calculate dynamic stops based on ATR if available
        dyn_stop, dyn_tp, dyn_trail_trigger = self.calculate_dynamic_stops(row, signal, row['close'])
        
        # If using enhanced exits and have Fibonacci or dynamic levels, use them
        if self.enhanced_exits and (fib_stop is not None or dyn_stop is not None):
            # Store original parameters
            original_sl_points = self.stop_loss_points
            original_tp_points = self.take_profit_points
            
            # Choose stop loss (prioritize Fibonacci stop if available, otherwise dynamic)
            chosen_stop = fib_stop if fib_stop is not None else dyn_stop
            
            # Multiple take profit points are available
            self.using_multiple_targets = self.use_multiple_targets and fib_tp1 is not None and fib_tp2 is not None
            
            # Calculate new stop loss and target points
            if signal == 'buy':
                sl_distance = abs(row['close'] - chosen_stop)
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # Set up multiple targets if enabled
                if self.using_multiple_targets:
                    self.tp_targets = [fib_tp1, fib_tp2, fib_tp3] if fib_tp3 is not None else [fib_tp1, fib_tp2]
                    print(f"Using multiple take profit targets: {[f'{tp:.3f}' for tp in self.tp_targets]}")
                    
                    # Use first target as primary take profit
                    tp_distance = abs(fib_tp1 - row['close'])
                    tp_points = int(tp_distance * 1000)
                else:
                    # Use Fibonacci or dynamic take profit
                    chosen_tp = fib_tp1 if fib_tp1 is not None else dyn_tp
                    tp_distance = abs(chosen_tp - row['close'])
                    tp_points = int(tp_distance * 1000)
                
                # Set dynamic trailing stop trigger price if available
                if dyn_trail_trigger is not None:
                    self.trailing_trigger_price = dyn_trail_trigger
                    print(f"Trailing stop will activate at {self.trailing_trigger_price:.3f}")
                
                # Only use if points are sufficiently large and respect min R:R
                r_r_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                if r_r_ratio >= self.r_r_ratio and tp_points >= 20 and sl_points >= 10:
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
                sl_distance = abs(chosen_stop - row['close'])
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # Set up multiple targets if enabled
                if self.using_multiple_targets:
                    self.tp_targets = [fib_tp1, fib_tp2, fib_tp3] if fib_tp3 is not None else [fib_tp1, fib_tp2]
                    print(f"Using multiple take profit targets: {[f'{tp:.3f}' for tp in self.tp_targets]}")
                    
                    # Use first target as primary take profit
                    tp_distance = abs(row['close'] - fib_tp1)
                    tp_points = int(tp_distance * 1000)
                else:
                    # Use Fibonacci or dynamic take profit
                    chosen_tp = fib_tp1 if fib_tp1 is not None else dyn_tp
                    tp_distance = abs(row['close'] - chosen_tp)
                    tp_points = int(tp_distance * 1000)
                
                # Set dynamic trailing stop trigger price if available
                if dyn_trail_trigger is not None:
                    self.trailing_trigger_price = dyn_trail_trigger
                    print(f"Trailing stop will activate at {self.trailing_trigger_price:.3f}")
                
                # Only use if points are sufficiently large and respect min R:R
                r_r_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                if r_r_ratio >= self.r_r_ratio and tp_points >= 20 and sl_points >= 10:
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
    
    def check_partial_take_profit(self, row):
        """
        Check if partial take profit should be executed
        
        Enhanced to work with multiple take profit targets
        """
        if not hasattr(self, 'using_multiple_targets') or not self.using_multiple_targets or not hasattr(self, 'tp_targets') or len(self.tp_targets) == 0:
            # Fall back to standard partial take profit logic if no multiple targets
            return super().check_partial_take_profit(row)
        
        if not self.position or self.partial_exit_done:
            return False, 0
            
        # Check if price hit the first target
        if self.position == 'buy':
            if hasattr(self, 'tp_targets') and len(self.tp_targets) > 0 and row["bid"] >= self.tp_targets[0]:
                # Calculate profit for partial position (33% at first target)
                partial_lot = self.position_size * 0.33
                points = (row["bid"] - self.entry_price) * 1000
                profit = points * partial_lot * 100
                
                # Reduce position size
                self.position_size -= partial_lot
                self.partial_exit_done = True
                self.partial_exits_count += 1
                
                print(f"First TP hit: {partial_lot:.2f} lots @ {row['bid']:.3f}, profit: {profit/100:.2f}$")
                
                # Move stop loss to breakeven
                self.stop_loss_price = self.entry_price
                print(f"Stop loss moved to breakeven: {self.stop_loss_price:.3f}")
                
                return True, profit
                
        elif self.position == 'sell':
            if hasattr(self, 'tp_targets') and len(self.tp_targets) > 0 and row["ask"] <= self.tp_targets[0]:
                # Calculate profit for partial position (33% at first target)
                partial_lot = self.position_size * 0.33
                points = (self.entry_price - row["ask"]) * 1000
                profit = points * partial_lot * 100
                
                # Reduce position size
                self.position_size -= partial_lot
                self.partial_exit_done = True
                self.partial_exits_count += 1
                
                print(f"First TP hit: {partial_lot:.2f} lots @ {row['ask']:.3f}, profit: {profit/100:.2f}$")
                
                # Move stop loss to breakeven
                self.stop_loss_price = self.entry_price
                print(f"Stop loss moved to breakeven: {self.stop_loss_price:.3f}")
                
                return True, profit
                
        return False, 0
    
    def check_exit_conditions(self, row):
        """
        Check if trade should be closed based on various exit conditions
        
        Enhanced for trailing stops and multiple take profit targets
        """
        if self.position is None:
            return False, 0, ""
        
        # Ensure all attributes are initialized
        if not hasattr(self, 'trailing_stop_activated'):
            self.trailing_stop_activated = False
        if not hasattr(self, 'trailing_trigger_price'):
            self.trailing_trigger_price = None
        if not hasattr(self, 'partial_exit_done'):
            self.partial_exit_done = False
        if not hasattr(self, 'tp_targets'):
            self.tp_targets = []
        if not hasattr(self, 'using_multiple_targets'):
            self.using_multiple_targets = False
        if not hasattr(self, 'trailing_stops_activated'):
            self.trailing_stops_activated = 0
        if not hasattr(self, 'partial_exits_profit'):
            self.partial_exits_profit = 0
            
        # First check for partial take profit at first target
        partial_hit, partial_profit = self.check_partial_take_profit(row)
        if partial_hit:
            # Update account but don't close position
            self.current_balance += partial_profit
            self.daily_profit_loss += partial_profit
            self.partial_exits_profit += partial_profit
        
        # Check trailing stop activation if target not yet hit
        try:
            if self.trailing_trigger_price is not None and not self.trailing_stop_activated:
                if self.position == 'buy' and row["bid"] >= self.trailing_trigger_price:
                    self.trailing_stop_activated = True
                    self.trailing_stops_activated += 1
                    print(f"Trailing stop activated at {row['bid']:.3f}")
                elif self.position == 'sell' and row["ask"] <= self.trailing_trigger_price:
                    self.trailing_stop_activated = True
                    self.trailing_stops_activated += 1
                    print(f"Trailing stop activated at {row['ask']:.3f}")
        except Exception as e:
            print(f"Warning: Error in trailing stop activation: {e}")
            self.trailing_trigger_price = None
        
        # Standard exit conditions with trailing stop handling
        c_bid = row["bid"]
        c_ask = row["ask"]
        
        if self.position == "buy":
            # Stop loss (includes trailing stop if activated)
            if c_bid <= self.stop_loss_price:
                points = (c_bid - self.entry_price) * 1000
                profit = points * self.position_size * 100
                return True, profit, "stop loss"
                
            # Second take profit target if partial exit done
            if self.using_multiple_targets and self.partial_exit_done:
                if len(self.tp_targets) > 1 and c_bid >= self.tp_targets[1]:
                    points = (c_bid - self.entry_price) * 1000
                    profit = points * self.position_size * 100
                    return True, profit, "take profit 2"
            
            # Regular take profit
            if c_bid >= self.take_profit_price:
                points = (c_bid - self.entry_price) * 1000
                profit = points * self.position_size * 100
                return True, profit, "take profit"
                
            # Update trailing stop if activated
            if self.trailing_stop_activated:
                new_stop = c_bid - self.trailing_distance * 0.001
                if new_stop > self.stop_loss_price:
                    self.stop_loss_price = new_stop
                    print(f"Trailing stop updated to {self.stop_loss_price:.3f}")

        else:  # sell
            # Stop loss (includes trailing stop if activated)
            if c_ask >= self.stop_loss_price:
                points = (self.entry_price - c_ask) * 1000
                profit = points * self.position_size * 100
                return True, profit, "stop loss"
                
            # Second take profit target if partial exit done
            if self.using_multiple_targets and self.partial_exit_done:
                if len(self.tp_targets) > 1 and c_ask <= self.tp_targets[1]:
                    points = (self.entry_price - c_ask) * 1000
                    profit = points * self.position_size * 100
                    return True, profit, "take profit 2"
            
            # Regular take profit
            if c_ask <= self.take_profit_price:
                points = (self.entry_price - c_ask) * 1000
                profit = points * self.position_size * 100
                return True, profit, "take profit"
                
            # Update trailing stop if activated
            if self.trailing_stop_activated:
                new_stop = c_ask + self.trailing_distance * 0.001
                if new_stop < self.stop_loss_price:
                    self.stop_loss_price = new_stop
                    print(f"Trailing stop updated to {self.stop_loss_price:.3f}")

        # Check time-based exit
        if self.check_time_based_exit(row.name):
            # Calculate current profit/loss
            if self.position == "buy":
                points = (c_bid - self.entry_price) * 1000
            else:  # sell
                points = (self.entry_price - c_ask) * 1000
                
            profit = points * self.position_size * 100
            return True, profit, "time exit"
        
        return False, 0, ""   

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
        self.trailing_stop_activated = False
        self.using_multiple_targets = False
        
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
        self.trend_filtered_signals = 0
        self.partial_exits_count = 0
        self.partial_exits_profit = 0
        self.trailing_stops_activated = 0
        
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
            results['trend_filtered_signals'] = self.trend_filtered_signals
            results['partial_exits_count'] = self.partial_exits_count
            results['partial_exits_profit'] = self.partial_exits_profit / 100  # Convert to USD
            results['trailing_stops_activated'] = self.trailing_stops_activated
            
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
            print(f"Fibonacci filtered signals: {self.filtered_signals}")
            print(f"Trend filtered signals: {self.trend_filtered_signals}")
            print(f"Partial exits: {self.partial_exits_count} (${self.partial_exits_profit/100:.2f} profit)")
            print(f"Trailing stops activated: {self.trailing_stops_activated}")
            
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
        hybrid_report = "\n\nEnhanced Hybrid Strategy Analysis\n" + "-" * 50 + "\n"
        hybrid_report += f"Hybrid strategy parameters:\n"
        hybrid_report += f"- Fibonacci analysis: {self.use_fibonacci}\n"
        hybrid_report += f"- Fibonacci lookback: {self.fibonacci_lookback}\n"
        hybrid_report += f"- Swing point lookback: {self.swing_lookback}\n"
        hybrid_report += f"- Fibonacci trend window: {self.fib_trend_window}\n"
        hybrid_report += f"- Prioritize 0.382 level: {self.prioritize_level_382}\n"
        hybrid_report += f"- Prioritize 0.618 level: {self.prioritize_level_618}\n"
        hybrid_report += f"- Fibonacci filter only: {self.fibonacci_filter_only}\n"
        hybrid_report += f"- Trend confirmation: {self.trend_confirmation}\n"
        hybrid_report += f"- Enhanced exits: {self.enhanced_exits}\n"
        hybrid_report += f"- Use multiple targets: {self.use_multiple_targets}\n"
        hybrid_report += f"- Trailing exit factor: {self.trailing_exit_factor}\n\n"
        
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
            hybrid_report += f"Fibonacci filtered signals: {self.filtered_signals}\n"
            hybrid_report += f"Trend filtered signals: {self.trend_filtered_signals}\n"
            hybrid_report += f"Partial exits: {self.partial_exits_count} (${self.partial_exits_profit/100:.2f} profit)\n"
            hybrid_report += f"Trailing stops activated: {self.trailing_stops_activated}\n\n"
        
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