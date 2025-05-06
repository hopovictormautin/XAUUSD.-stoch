"""
Candlestick Pattern Recognition Module for XAUUSD Trading Bot.

This module provides functions to identify Japanese candlestick patterns
and generate trading signals based on price action.
"""

import pandas as pd
import numpy as np


def identify_patterns(df, use_percentage=True, min_body_pct=1.0, max_body_pct=10.0):
    """
    Identify candlestick patterns in price data and add signals to DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Price data with OHLCV columns
    use_percentage : bool
        Whether to use percentage or point-based calculations for pattern identification
    min_body_pct : float
        Minimum candle body size as percentage of price for valid trades
    max_body_pct : float
        Maximum candle body size as percentage of price for valid trades
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added pattern columns and signals
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Calculate basic candle properties
    result['body_size'] = abs(result['close'] - result['open'])
    result['candle_range'] = result['high'] - result['low']
    result['body_pct'] = (result['body_size'] / result['close']) * 100
    result['is_bullish'] = result['close'] > result['open']
    result['is_bearish'] = result['close'] < result['open']
    
    # Filter for valid trading candles based on body size
    result['valid_trade'] = (result['body_pct'] >= min_body_pct) & (result['body_pct'] <= max_body_pct)
    
    # Add columns to track upper and lower shadows
    result['upper_shadow'] = result['high'] - result.apply(lambda x: max(x['open'], x['close']), axis=1)
    result['lower_shadow'] = result.apply(lambda x: min(x['open'], x['close']), axis=1) - result['low']
    
    # Initialize pattern signals columns
    pattern_columns = [
        'marubozu_bullish', 'marubozu_bearish', 
        'spinning_top', 'doji',
        'hammer', 'hanging_man', 'shooting_star',
        'engulfing_bullish', 'engulfing_bearish',
        'harami_bullish', 'harami_bearish',
        'piercing', 'dark_cloud',
        'morning_star', 'evening_star'
    ]
    
    for col in pattern_columns:
        result[col] = False
    
    # Initialize signal and stop loss columns
    result['pattern_signal'] = None  # 'buy', 'sell', or None
    result['pattern_stop_loss'] = None
    result['pattern_target'] = None
    result['pattern_name'] = None
    
    # Identify single candle patterns
    identify_single_candle_patterns(result)
    
    # Identify two candle patterns
    identify_two_candle_patterns(result)
    
    # Identify three candle patterns
    identify_three_candle_patterns(result)
    
    # Generate signals based on patterns
    generate_signals(result)
    
    return result


def identify_single_candle_patterns(df):
    """
    Identify single-candle patterns: Marubozu, Doji, Hammer, etc.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with candle properties already calculated
    """
    # Helper to check if values are approximately equal
    def approx_equal(a, b, threshold=0.05):
        return abs(a - b) <= threshold * max(abs(a), abs(b))
    
    # 1. Marubozu (strong trend candle)
    # Bullish Marubozu: Open ≈ Low AND Close ≈ High
    df['marubozu_bullish'] = (
        df['is_bullish'] & 
        df['valid_trade'] &
        (df['lower_shadow'] / df['candle_range'] < 0.1) &
        (df['upper_shadow'] / df['candle_range'] < 0.1) &
        (df['body_size'] / df['candle_range'] > 0.8)
    )
    
    # Bearish Marubozu: Open ≈ High AND Close ≈ Low
    df['marubozu_bearish'] = (
        df['is_bearish'] & 
        df['valid_trade'] &
        (df['lower_shadow'] / df['candle_range'] < 0.1) &
        (df['upper_shadow'] / df['candle_range'] < 0.1) &
        (df['body_size'] / df['candle_range'] > 0.8)
    )
    
    # 2. Spinning Top & Doji (indecision candles)
    # Small body + upper & lower shadows of approximately equal length
    df['spinning_top'] = (
        df['valid_trade'] &
        (df['body_size'] / df['candle_range'] < 0.3) &
        (df['body_size'] / df['candle_range'] > 0.1) &
        (abs(df['upper_shadow'] - df['lower_shadow']) / df['candle_range'] < 0.3)
    )
    
    df['doji'] = (
        df['valid_trade'] &
        (df['body_size'] / df['candle_range'] < 0.1) &
        (abs(df['upper_shadow'] - df['lower_shadow']) / df['candle_range'] < 0.3)
    )
    
    # 3. Hammer & Hanging Man
    # Lower shadow ≥ 2× body length
    umbrella_condition = (
        df['valid_trade'] &
        (df['lower_shadow'] >= 2 * df['body_size']) &
        (df['upper_shadow'] / df['candle_range'] < 0.1) &
        (df['lower_shadow'] / df['candle_range'] > 0.6)
    )
    
    # Hammer occurs at bottom of downtrend
    # Note: We'll need to identify trends separately for accurate identification
    # For now, we'll just identify the pattern
    df['hammer'] = umbrella_condition
    
    # Hanging Man occurs at top of uptrend
    df['hanging_man'] = umbrella_condition
    
    # 4. Shooting Star
    # Upper shadow ≥ 2× body length, small body
    df['shooting_star'] = (
        df['valid_trade'] &
        (df['upper_shadow'] >= 2 * df['body_size']) &
        (df['lower_shadow'] / df['candle_range'] < 0.1) &
        (df['upper_shadow'] / df['candle_range'] > 0.6)
    )


def identify_two_candle_patterns(df):
    """
    Identify two-candle patterns: Engulfing, Harami, Piercing, Dark Cloud Cover
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with candle properties already calculated
    """
    # 1. Bullish Engulfing
    for i in range(1, len(df)):
        # Bullish Engulfing: Day 2 blue body completely engulfs Day 1 red body
        if (df.iloc[i]['is_bullish'] and 
            df.iloc[i-1]['is_bearish'] and 
            df.iloc[i]['open'] <= df.iloc[i-1]['close'] and 
            df.iloc[i]['close'] >= df.iloc[i-1]['open'] and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-1]['valid_trade']):
            df.iloc[i, df.columns.get_loc('engulfing_bullish')] = True
        
        # Bearish Engulfing: Day 2 red body engulfs Day 1 blue body
        if (df.iloc[i]['is_bearish'] and 
            df.iloc[i-1]['is_bullish'] and 
            df.iloc[i]['open'] >= df.iloc[i-1]['close'] and 
            df.iloc[i]['close'] <= df.iloc[i-1]['open'] and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-1]['valid_trade']):
            df.iloc[i, df.columns.get_loc('engulfing_bearish')] = True
            
        # Bullish Harami: Day 2 small blue body contained within Day 1 red body
        if (df.iloc[i]['is_bullish'] and 
            df.iloc[i-1]['is_bearish'] and 
            df.iloc[i]['open'] >= df.iloc[i-1]['close'] and 
            df.iloc[i]['close'] <= df.iloc[i-1]['open'] and
            df.iloc[i]['body_size'] < df.iloc[i-1]['body_size'] * 0.6 and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-1]['valid_trade']):
            df.iloc[i, df.columns.get_loc('harami_bullish')] = True
            
        # Bearish Harami: Day 2 small red body within Day 1 blue body
        if (df.iloc[i]['is_bearish'] and 
            df.iloc[i-1]['is_bullish'] and 
            df.iloc[i]['open'] <= df.iloc[i-1]['close'] and 
            df.iloc[i]['close'] >= df.iloc[i-1]['open'] and
            df.iloc[i]['body_size'] < df.iloc[i-1]['body_size'] * 0.6 and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-1]['valid_trade']):
            df.iloc[i, df.columns.get_loc('harami_bearish')] = True
            
        # Piercing: Like bullish engulfing but Day 2 blue body covers 50–100% of Day 1 red body
        if (df.iloc[i]['is_bullish'] and 
            df.iloc[i-1]['is_bearish'] and 
            df.iloc[i]['open'] <= df.iloc[i-1]['close'] and 
            df.iloc[i]['close'] > df.iloc[i-1]['close'] + (df.iloc[i-1]['open'] - df.iloc[i-1]['close']) * 0.5 and
            df.iloc[i]['close'] < df.iloc[i-1]['open'] and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-1]['valid_trade']):
            df.iloc[i, df.columns.get_loc('piercing')] = True
            
        # Dark Cloud Cover: Like bearish engulfing but Day 2 red body covers 50–100% of Day 1 blue body
        if (df.iloc[i]['is_bearish'] and 
            df.iloc[i-1]['is_bullish'] and 
            df.iloc[i]['open'] >= df.iloc[i-1]['close'] and 
            df.iloc[i]['close'] < df.iloc[i-1]['open'] + (df.iloc[i-1]['close'] - df.iloc[i-1]['open']) * 0.5 and
            df.iloc[i]['close'] > df.iloc[i-1]['open'] and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-1]['valid_trade']):
            df.iloc[i, df.columns.get_loc('dark_cloud')] = True


def identify_three_candle_patterns(df):
    """
    Identify three-candle patterns: Morning Star, Evening Star
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with candle properties already calculated
    """
    for i in range(2, len(df)):
        # Morning Star: P1: long red; P2: gap-down + Doji/Spinning; P3: gap-up + blue closes above P1 midpoint
        if (df.iloc[i-2]['is_bearish'] and 
            (df.iloc[i-1]['spinning_top'] or df.iloc[i-1]['doji']) and
            df.iloc[i]['is_bullish'] and
            df.iloc[i-1]['high'] < df.iloc[i-2]['close'] and  # gap down
            df.iloc[i-1]['low'] > df.iloc[i-2]['low'] and
            df.iloc[i]['open'] > df.iloc[i-1]['high'] and  # gap up
            df.iloc[i]['close'] > (df.iloc[i-2]['open'] + df.iloc[i-2]['close']) / 2 and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-2]['valid_trade']):
            df.iloc[i, df.columns.get_loc('morning_star')] = True
            
        # Evening Star: P1: long blue; P2: gap-up + Doji/Spinning; P3: gap-down + red closes below P1 midpoint
        if (df.iloc[i-2]['is_bullish'] and 
            (df.iloc[i-1]['spinning_top'] or df.iloc[i-1]['doji']) and
            df.iloc[i]['is_bearish'] and
            df.iloc[i-1]['low'] > df.iloc[i-2]['close'] and  # gap up
            df.iloc[i-1]['high'] < df.iloc[i-2]['high'] and
            df.iloc[i]['open'] < df.iloc[i-1]['low'] and  # gap down
            df.iloc[i]['close'] < (df.iloc[i-2]['open'] + df.iloc[i-2]['close']) / 2 and
            df.iloc[i]['valid_trade'] and 
            df.iloc[i-2]['valid_trade']):
            df.iloc[i, df.columns.get_loc('evening_star')] = True


def generate_signals(df):
    """
    Generate trading signals based on identified patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with pattern columns
    """
    pattern_priority = [
        # Three-candle patterns (highest priority)
        'morning_star', 'evening_star',
        # Two-candle patterns
        'engulfing_bullish', 'engulfing_bearish',
        'harami_bullish', 'harami_bearish',
        'piercing', 'dark_cloud',
        # Single-candle patterns (lowest priority)
        'marubozu_bullish', 'marubozu_bearish',
        'hammer', 'hanging_man', 'shooting_star',
        'spinning_top', 'doji'
    ]
    
    # Define pattern properties (signal type and stop loss calculation)
    pattern_properties = {
        'morning_star': {'signal': 'buy', 'stop_method': 'lowest_low_3', 'pattern_name': 'Morning Star'},
        'evening_star': {'signal': 'sell', 'stop_method': 'highest_high_3', 'pattern_name': 'Evening Star'},
        'engulfing_bullish': {'signal': 'buy', 'stop_method': 'lowest_low_2', 'pattern_name': 'Bullish Engulfing'},
        'engulfing_bearish': {'signal': 'sell', 'stop_method': 'highest_high_2', 'pattern_name': 'Bearish Engulfing'},
        'harami_bullish': {'signal': 'buy', 'stop_method': 'lowest_low_2', 'pattern_name': 'Bullish Harami'},
        'harami_bearish': {'signal': 'sell', 'stop_method': 'highest_high_2', 'pattern_name': 'Bearish Harami'},
        'piercing': {'signal': 'buy', 'stop_method': 'lowest_low_2', 'pattern_name': 'Piercing Line'},
        'dark_cloud': {'signal': 'sell', 'stop_method': 'highest_high_2', 'pattern_name': 'Dark Cloud Cover'},
        'marubozu_bullish': {'signal': 'buy', 'stop_method': 'bar_low', 'pattern_name': 'Bullish Marubozu'},
        'marubozu_bearish': {'signal': 'sell', 'stop_method': 'bar_high', 'pattern_name': 'Bearish Marubozu'},
        'hammer': {'signal': 'buy', 'stop_method': 'bar_low', 'pattern_name': 'Hammer'},
        'hanging_man': {'signal': 'sell', 'stop_method': 'bar_high', 'pattern_name': 'Hanging Man'},
        'shooting_star': {'signal': 'sell', 'stop_method': 'bar_high', 'pattern_name': 'Shooting Star'},
        'spinning_top': {'signal': None, 'stop_method': None, 'pattern_name': 'Spinning Top'},
        'doji': {'signal': None, 'stop_method': None, 'pattern_name': 'Doji'}
    }
    
    # Assign signals based on pattern priority
    for i in range(len(df)):
        for pattern in pattern_priority:
            if df.iloc[i][pattern]:
                props = pattern_properties[pattern]
                
                # Skip patterns that don't generate signals
                if props['signal'] is None:
                    continue
                
                df.iloc[i, df.columns.get_loc('pattern_signal')] = props['signal']
                df.iloc[i, df.columns.get_loc('pattern_name')] = props['pattern_name']
                
                # Calculate stop loss based on pattern type
                if props['stop_method'] == 'bar_low':
                    df.iloc[i, df.columns.get_loc('pattern_stop_loss')] = df.iloc[i]['low']
                elif props['stop_method'] == 'bar_high':
                    df.iloc[i, df.columns.get_loc('pattern_stop_loss')] = df.iloc[i]['high']
                elif props['stop_method'] == 'lowest_low_2':
                    if i > 0:
                        df.iloc[i, df.columns.get_loc('pattern_stop_loss')] = min(df.iloc[i]['low'], df.iloc[i-1]['low'])
                elif props['stop_method'] == 'highest_high_2':
                    if i > 0:
                        df.iloc[i, df.columns.get_loc('pattern_stop_loss')] = max(df.iloc[i]['high'], df.iloc[i-1]['high'])
                elif props['stop_method'] == 'lowest_low_3':
                    if i > 1:
                        df.iloc[i, df.columns.get_loc('pattern_stop_loss')] = min(df.iloc[i]['low'], df.iloc[i-1]['low'], df.iloc[i-2]['low'])
                elif props['stop_method'] == 'highest_high_3':
                    if i > 1:
                        df.iloc[i, df.columns.get_loc('pattern_stop_loss')] = max(df.iloc[i]['high'], df.iloc[i-1]['high'], df.iloc[i-2]['high'])
                
                # Only use the highest priority pattern
                break


def find_support_resistance(df, lookback=100, threshold=0.01):
    """
    Identify horizontal support and resistance levels from price action
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Price data with OHLC columns
    lookback : int
        Number of bars to look back for S/R identification
    threshold : float
        Percentage threshold for price clusters
        
    Returns:
    --------
    list
        List of support and resistance levels
    """
    price_points = []
    
    # Collect significant price points (highs, lows, and where price reversed)
    for i in range(1, min(lookback, len(df)-1)):
        # Significant highs (price reversed from this level)
        if df.iloc[i]['high'] > df.iloc[i-1]['high'] and df.iloc[i]['high'] > df.iloc[i+1]['high']:
            price_points.append(df.iloc[i]['high'])
        
        # Significant lows (price reversed from this level)
        if df.iloc[i]['low'] < df.iloc[i-1]['low'] and df.iloc[i]['low'] < df.iloc[i+1]['low']:
            price_points.append(df.iloc[i]['low'])
    
    # Group price points into clusters
    levels = []
    price_points.sort()
    
    current_cluster = [price_points[0]] if price_points else []
    
    for price in price_points[1:]:
        # If price is within threshold of current cluster's mean
        if price <= current_cluster[-1] * (1 + threshold):
            current_cluster.append(price)
        else:
            # Found a new level, save mean of previous cluster
            if len(current_cluster) >= 3:  # Need at least 3 touches to confirm level
                levels.append(sum(current_cluster) / len(current_cluster))
            # Start a new cluster
            current_cluster = [price]
    
    # Don't forget the last cluster
    if len(current_cluster) >= 3:
        levels.append(sum(current_cluster) / len(current_cluster))
    
    return levels


def set_profit_targets(df, levels, lookback=50):
    """
    Set profit targets based on support/resistance levels
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with signals
    levels : list
        List of support and resistance levels
    lookback : int
        How far back to look for levels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added profit target column
    """
    result = df.copy()
    
    for i in range(len(result)):
        if result.iloc[i]['pattern_signal'] == 'buy':
            # For buy signals, find next resistance level above entry
            entry_price = result.iloc[i]['close']
            targets = [level for level in levels if level > entry_price]
            
            if targets:
                # Choose closest resistance level
                target = min(targets)
                result.iloc[i, result.columns.get_loc('pattern_target')] = target
                
        elif result.iloc[i]['pattern_signal'] == 'sell':
            # For sell signals, find next support level below entry
            entry_price = result.iloc[i]['close']
            targets = [level for level in levels if level < entry_price]
            
            if targets:
                # Choose closest support level
                target = max(targets)
                result.iloc[i, result.columns.get_loc('pattern_target')] = target
    
    return result


def calculate_trade_risk_reward(df):
    """
    Calculate risk and potential reward for each signal
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with signals, stops, and targets
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added risk-reward metrics
    """
    result = df.copy()
    
    result['risk_points'] = None
    result['reward_points'] = None
    result['risk_reward_ratio'] = None
    
    for i in range(len(result)):
        if result.iloc[i]['pattern_signal'] == 'buy' and result.iloc[i]['pattern_stop_loss'] is not None:
            entry = result.iloc[i]['close']
            stop = result.iloc[i]['pattern_stop_loss']
            target = result.iloc[i]['pattern_target']
            
            risk = entry - stop
            
            if target is not None:
                reward = target - entry
                if risk > 0 and reward > 0:
                    result.iloc[i, result.columns.get_loc('risk_points')] = risk
                    result.iloc[i, result.columns.get_loc('reward_points')] = reward
                    result.iloc[i, result.columns.get_loc('risk_reward_ratio')] = reward / risk
            
        elif result.iloc[i]['pattern_signal'] == 'sell' and result.iloc[i]['pattern_stop_loss'] is not None:
            entry = result.iloc[i]['close']
            stop = result.iloc[i]['pattern_stop_loss']
            target = result.iloc[i]['pattern_target']
            
            risk = stop - entry
            
            if target is not None:
                reward = entry - target
                if risk > 0 and reward > 0:
                    result.iloc[i, result.columns.get_loc('risk_points')] = risk
                    result.iloc[i, result.columns.get_loc('reward_points')] = reward
                    result.iloc[i, result.columns.get_loc('risk_reward_ratio')] = reward / risk
    
    return result


def combine_with_indicator_signals(candlestick_df, indicator_signals):
    """
    Combine candlestick pattern signals with technical indicator signals
    
    Parameters:
    -----------
    candlestick_df : pandas.DataFrame
        DataFrame with candlestick pattern signals
    indicator_signals : pandas.DataFrame
        DataFrame with technical indicator signals
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with combined signals
    """
    # Merge dataframes on index (assuming both have same index)
    combined = candlestick_df.copy()
    
    # Create combined signal column
    combined['combined_signal'] = None
    
    # Logic for combining signals
    for i in range(len(combined)):
        # If both agree, take the signal
        if (combined.iloc[i]['pattern_signal'] == 'buy' and indicator_signals.iloc[i]['signal'] == 'buy'):
            combined.iloc[i, combined.columns.get_loc('combined_signal')] = 'buy'
        elif (combined.iloc[i]['pattern_signal'] == 'sell' and indicator_signals.iloc[i]['signal'] == 'sell'):
            combined.iloc[i, combined.columns.get_loc('combined_signal')] = 'sell'
        # If pattern has a signal but indicators don't, take pattern signal with reduced position size
        elif combined.iloc[i]['pattern_signal'] is not None:
            combined.iloc[i, combined.columns.get_loc('combined_signal')] = combined.iloc[i]['pattern_signal'] + '_half'
    
    return combined