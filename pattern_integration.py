"""
Integration module for combining XAUUSD bot with candlestick pattern analysis.

This module provides functions to:
1. Merge signals from existing indicators with candlestick patterns
2. Adapt position sizing based on pattern confidence
3. Adjust stop loss and take profit levels based on pattern recognition
"""

import pandas as pd
import numpy as np
from utils.candlestick_patterns import identify_patterns, find_support_resistance, set_profit_targets, calculate_trade_risk_reward

def enhance_signals_with_patterns(bot, df):
    """
    Enhance trading signals with candlestick pattern recognition
    
    Parameters:
    -----------
    bot : XAUUSDCentScalpingBot
        Bot instance with base strategy
    df : pandas.DataFrame
        Price data with indicator signals

    Returns:
    --------
    pandas.DataFrame
        DataFrame with enhanced signals
    """
    # First identify candlestick patterns
    pattern_df = identify_patterns(df)
    
    # Find support and resistance levels
    levels = find_support_resistance(df, lookback=100)
    
    # Set profit targets based on support/resistance
    pattern_df = set_profit_targets(pattern_df, levels)
    
    # Calculate risk/reward for each signal
    pattern_df = calculate_trade_risk_reward(pattern_df)
    
    # Initialize combined signal columns
    pattern_df['combined_signal'] = None
    pattern_df['combined_stop_loss'] = None
    pattern_df['combined_target'] = None
    pattern_df['position_size_factor'] = 1.0  # Default multiplier
    
    # Combine signals from technical indicators and candlestick patterns
    for i in range(len(pattern_df)):
        # Check if we have a technical indicator signal (based on stochastic, MA, etc.)
        tech_signal = None
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns and 'fast_ma' in df.columns and 'slow_ma' in df.columns:
            # Buy signal from original strategy
            if (df.iloc[i]['fast_ma'] > df.iloc[i]['slow_ma'] and 
                df.iloc[i]['stoch_k'] < bot.stoch_lower_level and 
                df.iloc[i]['stoch_k'] > df.iloc[i]['stoch_d']):
                tech_signal = 'buy'
                
            # Sell signal from original strategy
            elif (df.iloc[i]['fast_ma'] < df.iloc[i]['slow_ma'] and 
                  df.iloc[i]['stoch_k'] > bot.stoch_upper_level and 
                  df.iloc[i]['stoch_k'] < df.iloc[i]['stoch_d']):
                tech_signal = 'sell'
        
        # Get pattern signal
        pattern_signal = pattern_df.iloc[i]['pattern_signal']
        
        # Combine signals based on confidence levels
        if tech_signal == pattern_signal and tech_signal is not None:
            # Both strategies agree - strongest signal
            pattern_df.iloc[i, pattern_df.columns.get_loc('combined_signal')] = tech_signal
            pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = 1.0  # Full position
            
            # Use more conservative stop loss (further from entry)
            if tech_signal == 'buy':
                tech_stop = df.iloc[i]['low'] - (0.1 * df.iloc[i]['atr']) if 'atr' in df.columns else df.iloc[i]['low']
                pattern_stop = pattern_df.iloc[i]['pattern_stop_loss']
                if pattern_stop is not None and tech_stop is not None:
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = min(tech_stop, pattern_stop)
                else:
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = pattern_stop or tech_stop
                    
                # Set profit target based on pattern and/or support/resistance
                if pattern_df.iloc[i]['pattern_target'] is not None:
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = pattern_df.iloc[i]['pattern_target']
                elif 'atr' in df.columns:
                    # If no pattern target, use ATR-based target (2:1 reward-risk ratio)
                    stop_distance = df.iloc[i]['close'] - (pattern_df.iloc[i]['combined_stop_loss'] or tech_stop)
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = df.iloc[i]['close'] + (stop_distance * 2)
                
            elif tech_signal == 'sell':
                tech_stop = df.iloc[i]['high'] + (0.1 * df.iloc[i]['atr']) if 'atr' in df.columns else df.iloc[i]['high']
                pattern_stop = pattern_df.iloc[i]['pattern_stop_loss']
                if pattern_stop is not None and tech_stop is not None:
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = max(tech_stop, pattern_stop)
                else:
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = pattern_stop or tech_stop
                    
                # Set profit target based on pattern and/or support/resistance
                if pattern_df.iloc[i]['pattern_target'] is not None:
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = pattern_df.iloc[i]['pattern_target']
                elif 'atr' in df.columns:
                    # If no pattern target, use ATR-based target (2:1 reward-risk ratio)
                    stop_distance = (pattern_df.iloc[i]['combined_stop_loss'] or tech_stop) - df.iloc[i]['close']
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = df.iloc[i]['close'] - (stop_distance * 2)
                    
        elif tech_signal is not None:
            # Only technical signal - normal position
            pattern_df.iloc[i, pattern_df.columns.get_loc('combined_signal')] = tech_signal
            pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = 0.8  # Reduced position (80%)
            
            # Use standard technical stops and targets
            if tech_signal == 'buy':
                tech_stop = df.iloc[i]['low'] - (0.1 * df.iloc[i]['atr']) if 'atr' in df.columns else df.iloc[i]['low']
                pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = tech_stop
                
                # Set ATR-based target
                if 'atr' in df.columns:
                    stop_distance = df.iloc[i]['close'] - tech_stop
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = df.iloc[i]['close'] + (stop_distance * bot.r_r_ratio)
                
            elif tech_signal == 'sell':
                tech_stop = df.iloc[i]['high'] + (0.1 * df.iloc[i]['atr']) if 'atr' in df.columns else df.iloc[i]['high']
                pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = tech_stop
                
                # Set ATR-based target
                if 'atr' in df.columns:
                    stop_distance = tech_stop - df.iloc[i]['close']
                    pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = df.iloc[i]['close'] - (stop_distance * bot.r_r_ratio)
                    
        elif pattern_signal is not None:
            # Only pattern signal - reduced position
            pattern_df.iloc[i, pattern_df.columns.get_loc('combined_signal')] = pattern_signal
            pattern_df.iloc[i, pattern_df.columns.get_loc('combined_stop_loss')] = pattern_df.iloc[i]['pattern_stop_loss']
            pattern_df.iloc[i, pattern_df.columns.get_loc('combined_target')] = pattern_df.iloc[i]['pattern_target']
            
            # Adjust position size based on pattern type
            pattern_name = pattern_df.iloc[i]['pattern_name']
            
            # Stronger patterns get larger position sizes
            if pattern_name in ['Morning Star', 'Evening Star', 'Bullish Engulfing', 'Bearish Engulfing']:
                pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = 0.75  # 75% position
            elif pattern_name in ['Piercing Line', 'Dark Cloud Cover', 'Bullish Harami', 'Bearish Harami']:
                pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = 0.5   # 50% position
            else:
                pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = 0.3   # 30% position
            
            # Adjust position size further based on risk-reward ratio
            if pattern_df.iloc[i]['risk_reward_ratio'] is not None:
                r_r = pattern_df.iloc[i]['risk_reward_ratio']
                if r_r >= 3.0:
                    # Increase position for excellent risk-reward
                    current_factor = pattern_df.iloc[i]['position_size_factor']
                    pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = min(1.0, current_factor * 1.2)
                elif r_r <= 1.0:
                    # Decrease position for poor risk-reward
                    current_factor = pattern_df.iloc[i]['position_size_factor']
                    pattern_df.iloc[i, pattern_df.columns.get_loc('position_size_factor')] = current_factor * 0.8
                    
    return pattern_df


def apply_pattern_strategy(bot, df):
    """
    Apply pattern recognition and modify bot's trading decisions
    
    Parameters:
    -----------
    bot : XAUUSDCentScalpingBot
        Bot instance
    df : pandas.DataFrame
        Price data with indicators
        
    Returns:
    --------
    tuple
        Enhanced signal, stop loss, take profit, position size factor
    """
    # Get enhanced signals
    enhanced_df = enhance_signals_with_patterns(bot, df)
    
    # Get latest bar
    latest = enhanced_df.iloc[-1]
    
    # Return signal details
    return (
        latest['combined_signal'],
        latest['combined_stop_loss'],
        latest['combined_target'],
        latest['position_size_factor']
    )


def integrate_patterns_with_bot(bot):
    """
    Integrate pattern recognition functionality into the bot's workflow
    
    Parameters:
    -----------
    bot : XAUUSDCentScalpingBot
        Bot instance to modify
        
    Returns:
    --------
    XAUUSDCentScalpingBot
        Modified bot with pattern recognition
    """
    # Store original check_entry_signals method
    original_check_entry = bot.check_entry_signals
    
    # Define new method that enhances the original
    def enhanced_check_entry_signals(self, row):
        # Get original signal first
        original_signal = original_check_entry(row)
        
        # If we have enough history, apply pattern recognition
        if len(self.prepared_data) >= 20:  # Need some history for patterns
            # Get last 50 rows of data (or all if less than 50)
            history = self.prepared_data.iloc[-min(50, len(self.prepared_data)):]
            
            pattern_signal, pattern_stop, pattern_target, size_factor = apply_pattern_strategy(self, history)
            
            # Store pattern info for position sizing and exit management
            self.pattern_stop_loss = pattern_stop
            self.pattern_take_profit = pattern_target
            self.position_size_factor = size_factor
            
            # If original strategy gives no signal but pattern does, use pattern
            if original_signal is None and pattern_signal is not None:
                print(f"Pattern signal generated: {pattern_signal}")
                return pattern_signal
                
            # If both strategies give signals and they conflict, use original (more conservative)
            if original_signal is not None and pattern_signal is not None and original_signal != pattern_signal:
                print(f"Signal conflict: Original={original_signal}, Pattern={pattern_signal}. Using original.")
                return original_signal
        
        return original_signal
    
    # Store original calculate_lot_size method
    original_calc_lot = bot.calculate_lot_size
    
    # Define enhanced method that adjusts position size based on pattern confidence
    def enhanced_calculate_lot_size(self, stop_loss_points):
        # Get base lot size from original method
        base_lot_size = original_calc_lot(stop_loss_points)
        
        # Adjust based on pattern confidence if available
        if hasattr(self, 'position_size_factor'):
            adjusted_lot_size = base_lot_size * self.position_size_factor
            print(f"Adjusting position size by factor {self.position_size_factor}: {base_lot_size} → {adjusted_lot_size}")
            return adjusted_lot_size
        
        return base_lot_size
    
    # Store original open_position method
    original_open_position = bot.open_position
    
    # Define enhanced method that uses pattern-based stops and targets when available
    def enhanced_open_position(self, signal, row, market_regime="normal_volatility"):
        # Check if we have pattern-based stops and targets
        if hasattr(self, 'pattern_stop_loss') and self.pattern_stop_loss is not None:
            print(f"Using pattern-based stop loss: {self.pattern_stop_loss}")
            
            # Calculate dynamic exits based on pattern recognition
            if signal == 'buy':
                sl_distance = abs(row['close'] - self.pattern_stop_loss)
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # Override default parameters for this trade
                original_sl = self.stop_loss_points
                self.stop_loss_points = sl_points
                
                # Call original method with modified parameters
                result = original_open_position(signal, row, market_regime)
                
                # Restore original parameters
                self.stop_loss_points = original_sl
                
                # Override take profit if pattern target exists
                if hasattr(self, 'pattern_take_profit') and self.pattern_take_profit is not None:
                    self.take_profit_price = self.pattern_take_profit
                    print(f"Using pattern-based take profit: {self.pattern_take_profit}")
                
                return result
                
            elif signal == 'sell':
                sl_distance = abs(self.pattern_stop_loss - row['close'])
                sl_points = int(sl_distance * 1000)  # Convert to points
                
                # Override default parameters for this trade
                original_sl = self.stop_loss_points
                self.stop_loss_points = sl_points
                
                # Call original method with modified parameters
                result = original_open_position(signal, row, market_regime)
                
                # Restore original parameters
                self.stop_loss_points = original_sl
                
                # Override take profit if pattern target exists
                if hasattr(self, 'pattern_take_profit') and self.pattern_take_profit is not None:
                    self.take_profit_price = self.pattern_take_profit
                    print(f"Using pattern-based take profit: {self.pattern_take_profit}")
                    
                return result
        
        # If no pattern-based stops, use original method
        return original_open_position(signal, row, market_regime)
    
    # Replace methods with enhanced versions
    bot.check_entry_signals = lambda row: enhanced_check_entry_signals(bot, row)
    bot.calculate_lot_size = lambda stop_loss_points: enhanced_calculate_lot_size(bot, stop_loss_points)
    bot.open_position = lambda signal, row, market_regime="normal_volatility": enhanced_open_position(bot, signal, row, market_regime)
    
    # Add storage for prepared data (needed for pattern recognition)
    bot.prepared_data = pd.DataFrame()
    
    # Enhance prepare_data method to store history
    original_prepare_data = bot.prepare_data
    
    def enhanced_prepare_data(self, df):
        result = original_prepare_data(df)
        self.prepared_data = result
        return result
    
    bot.prepare_data = lambda df: enhanced_prepare_data(bot, df)
    
    # Add pattern recognition attributes
    bot.pattern_stop_loss = None
    bot.pattern_take_profit = None
    bot.position_size_factor = 1.0
    
    return bot