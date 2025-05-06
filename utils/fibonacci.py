"""
Fibonacci Retracement and Extension Module for XAUUSD Trading Bot.

This module provides functions to identify Fibonacci levels, detect retracements,
and generate trading signals based on Fibonacci principles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Standard Fibonacci levels
FIBO_LEVELS = {
    "retracement": [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
    "extension": [0.0, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618]
}

def find_swing_points(df, lookback=10):
    """
    Find swing highs and lows in price data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC price data
    lookback : int
        Number of bars to look back/forward for swing point detection
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added swing point columns
    """
    result = df.copy()
    
    # Initialize swing point columns
    result['swing_high'] = False
    result['swing_low'] = False
    
    # Find swing highs
    for i in range(lookback, len(result) - lookback):
        # Check if current high is highest in the window
        if result.iloc[i]['high'] == result.iloc[i-lookback:i+lookback+1]['high'].max():
            result.iloc[i, result.columns.get_loc('swing_high')] = True
    
    # Find swing lows
    for i in range(lookback, len(result) - lookback):
        # Check if current low is lowest in the window
        if result.iloc[i]['low'] == result.iloc[i-lookback:i+lookback+1]['low'].min():
            result.iloc[i, result.columns.get_loc('swing_low')] = True
    
    return result

def detect_trend(df, window=20):
    """
    Detect current market trend
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC price data
    window : int
        Window size for trend detection
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added trend column
    """
    result = df.copy()
    
    # Use moving average direction for trend detection
    result['ma'] = result['close'].rolling(window).mean()
    result['trend'] = 'neutral'
    
    for i in range(window, len(result)):
        if result.iloc[i]['ma'] > result.iloc[i-window]['ma']:
            result.iloc[i, result.columns.get_loc('trend')] = 'uptrend'
        elif result.iloc[i]['ma'] < result.iloc[i-window]['ma']:
            result.iloc[i, result.columns.get_loc('trend')] = 'downtrend'
    
    return result

def calculate_fibonacci_levels(start_price, end_price, levels=None):
    """
    Calculate Fibonacci levels
    
    Parameters:
    -----------
    start_price : float
        Starting price point
    end_price : float
        Ending price point
    levels : list, optional
        Specific Fibonacci levels to calculate
        
    Returns:
    --------
    dict
        Dictionary of Fibonacci levels and their price values
    """
    if levels is None:
        levels = FIBO_LEVELS["retracement"]
    
    price_range = end_price - start_price
    return {level: start_price + price_range * (1 - level) if end_price > start_price
            else start_price - price_range * level for level in levels}

def identify_fibonacci_retracements(df, lookback=50, swing_lookback=10, trend_window=20, plot=False):
    """
    Identify Fibonacci retracement levels and potential trade signals
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC price data
    lookback : int
        Number of bars to look back for analysis
    swing_lookback : int
        Lookback for swing point detection
    trend_window : int
        Window for trend detection
    plot : bool
        Whether to plot the identified retracements
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Fibonacci signal columns
    """
    # Ensure we have enough data
    if len(df) < lookback + swing_lookback:
        print(f"Not enough data for Fibonacci analysis: {len(df)} rows, need at least {lookback + swing_lookback}")
        return df
    
    # Find swing points and detect trend
    result = find_swing_points(df, lookback=swing_lookback)
    result = detect_trend(result, window=trend_window)
    
    # Initialize columns
    result['fib_retracement'] = False
    result['fib_level_touched'] = None
    result['fib_start_price'] = None
    result['fib_end_price'] = None
    result['fib_signal'] = None
    
    # Look for retracement opportunities in recent data
    for i in range(lookback, len(result)):
        # Skip if we already have a signal for this bar
        if result.iloc[i]['fib_retracement']:
            continue
            
        # Get recent trend
        current_trend = result.iloc[i]['trend']
        
        if current_trend == 'uptrend':
            # In uptrend, look for recent swing low followed by swing high
            recent_data = result.iloc[i-lookback:i]
            swing_lows = recent_data[recent_data['swing_low']].index
            swing_highs = recent_data[recent_data['swing_high']].index
            
            if len(swing_lows) > 0 and len(swing_highs) > 0:
                # Get most recent swing points
                latest_low = swing_lows[-1]
                latest_high = swing_highs[-1]
                
                # Ensure swing low comes before swing high (proper sequence for uptrend)
                if latest_low < latest_high:
                    # Calculate Fibonacci levels from swing low to swing high
                    low_price = result.loc[latest_low, 'low']
                    high_price = result.loc[latest_high, 'high']
                    
                    fib_levels = calculate_fibonacci_levels(low_price, high_price)
                    
                    # Check if current price is near a Fibonacci retracement level
                    current_price = result.iloc[i]['close']
                    
                    # Identify which level (if any) is being tested
                    level_touched = None
                    for level, price in fib_levels.items():
                        # Use a small buffer (0.5% of price) for level detection
                        buffer = price * 0.005
                        if current_price >= price - buffer and current_price <= price + buffer:
                            level_touched = level
                            break
                    
                    # If a level is being tested and it's between 0.236 and 0.786 (not extremes)
                    if level_touched is not None and 0.236 <= level_touched <= 0.786:
                        result.iloc[i, result.columns.get_loc('fib_retracement')] = True
                        result.iloc[i, result.columns.get_loc('fib_level_touched')] = level_touched
                        result.iloc[i, result.columns.get_loc('fib_start_price')] = low_price
                        result.iloc[i, result.columns.get_loc('fib_end_price')] = high_price
                        
                        # Generate buy signal at Fibonacci support in uptrend
                        result.iloc[i, result.columns.get_loc('fib_signal')] = 'buy'
                        
        elif current_trend == 'downtrend':
            # In downtrend, look for recent swing high followed by swing low
            recent_data = result.iloc[i-lookback:i]
            swing_lows = recent_data[recent_data['swing_low']].index
            swing_highs = recent_data[recent_data['swing_high']].index
            
            if len(swing_lows) > 0 and len(swing_highs) > 0:
                # Get most recent swing points
                latest_high = swing_highs[-1]
                latest_low = swing_lows[-1]
                
                # Ensure swing high comes before swing low (proper sequence for downtrend)
                if latest_high < latest_low:
                    # Calculate Fibonacci levels from swing high to swing low
                    high_price = result.loc[latest_high, 'high']
                    low_price = result.loc[latest_low, 'low']
                    
                    fib_levels = calculate_fibonacci_levels(high_price, low_price)
                    
                    # Check if current price is near a Fibonacci retracement level
                    current_price = result.iloc[i]['close']
                    
                    # Identify which level (if any) is being tested
                    level_touched = None
                    for level, price in fib_levels.items():
                        # Use a small buffer (0.5% of price) for level detection
                        buffer = price * 0.005
                        if current_price >= price - buffer and current_price <= price + buffer:
                            level_touched = level
                            break
                    
                    # If a level is being tested and it's between 0.236 and 0.786 (not extremes)
                    if level_touched is not None and 0.236 <= level_touched <= 0.786:
                        result.iloc[i, result.columns.get_loc('fib_retracement')] = True
                        result.iloc[i, result.columns.get_loc('fib_level_touched')] = level_touched
                        result.iloc[i, result.columns.get_loc('fib_start_price')] = high_price
                        result.iloc[i, result.columns.get_loc('fib_end_price')] = low_price
                        
                        # Generate sell signal at Fibonacci resistance in downtrend
                        result.iloc[i, result.columns.get_loc('fib_signal')] = 'sell'
    
    # Plot if requested
    if plot and result['fib_retracement'].any():
        plot_fibonacci_levels(result)
    
    return result

def calculate_fibonacci_extensions(start_price, end_price, retracement_price):
    """
    Calculate Fibonacci extension levels
    
    Parameters:
    -----------
    start_price : float
        Starting price point (swing low in uptrend)
    end_price : float
        Ending price point (swing high in uptrend)
    retracement_price : float
        Price at retracement level
        
    Returns:
    --------
    dict
        Dictionary of Fibonacci extension levels and their price values
    """
    levels = FIBO_LEVELS["extension"]
    price_range = end_price - start_price
    
    extensions = {}
    for level in levels:
        if level <= 1.0:
            # For 0 to 100%, it's just the normal range from start to end
            extensions[level] = start_price + price_range * level
        else:
            # For extensions beyond 100%, continue the trend beyond the end price
            extensions[level] = end_price + price_range * (level - 1.0)
    
    return extensions

def set_fibonacci_targets(df, atr_factor=1.5):
    """
    Set stop loss and take profit targets based on Fibonacci levels
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Fibonacci signals
    atr_factor : float
        Multiplier for ATR-based stop loss buffer
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added target columns
    """
    result = df.copy()
    
    # Initialize target columns
    result['fib_stop_loss'] = None
    result['fib_take_profit1'] = None  # 61.8% extension
    result['fib_take_profit2'] = None  # 100% extension
    result['fib_take_profit3'] = None  # 161.8% extension
    
    # Calculate ATR for stop loss buffer
    if 'atr' not in result.columns:
        # Calculate ATR if not present
        high_low = result['high'] - result['low']
        high_close = abs(result['high'] - result['close'].shift())
        low_close = abs(result['low'] - result['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result['atr'] = true_range.rolling(14).mean()
    
    # Set targets for each signal
    for i in range(len(result)):
        if result.iloc[i]['fib_signal'] == 'buy':
            # For buy signals, calculate extension levels from start to end
            start_price = result.iloc[i]['fib_start_price']
            end_price = result.iloc[i]['fib_end_price']
            level_touched = result.iloc[i]['fib_level_touched']
            
            if start_price is not None and end_price is not None and level_touched is not None:
                # Calculate current retracement price
                retracement_price = result.iloc[i]['close']
                
                # Calculate extension levels
                extensions = calculate_fibonacci_extensions(start_price, end_price, retracement_price)
                
                # Set stop loss just below the retracement level with ATR buffer
                atr_buffer = result.iloc[i]['atr'] * atr_factor if 'atr' in result.columns else 0
                stop_level = min(level_touched + 0.146, 1.0)  # Next Fib level or 100%
                stop_price = calculate_fibonacci_levels(start_price, end_price, [stop_level])[stop_level]
                result.iloc[i, result.columns.get_loc('fib_stop_loss')] = stop_price - atr_buffer
                
                # Set take profit targets at extension levels
                result.iloc[i, result.columns.get_loc('fib_take_profit1')] = extensions[0.618]
                result.iloc[i, result.columns.get_loc('fib_take_profit2')] = extensions[1.0]
                result.iloc[i, result.columns.get_loc('fib_take_profit3')] = extensions[1.618]
                
        elif result.iloc[i]['fib_signal'] == 'sell':
            # For sell signals, calculate extension levels from start to end
            start_price = result.iloc[i]['fib_start_price']
            end_price = result.iloc[i]['fib_end_price']
            level_touched = result.iloc[i]['fib_level_touched']
            
            if start_price is not None and end_price is not None and level_touched is not None:
                # Calculate current retracement price
                retracement_price = result.iloc[i]['close']
                
                # Calculate extension levels
                extensions = calculate_fibonacci_extensions(start_price, end_price, retracement_price)
                
                # Set stop loss just above the retracement level with ATR buffer
                atr_buffer = result.iloc[i]['atr'] * atr_factor if 'atr' in result.columns else 0
                stop_level = min(level_touched + 0.146, 1.0)  # Next Fib level or 100%
                stop_price = calculate_fibonacci_levels(start_price, end_price, [stop_level])[stop_level]
                result.iloc[i, result.columns.get_loc('fib_stop_loss')] = stop_price + atr_buffer
                
                # Set take profit targets at extension levels
                result.iloc[i, result.columns.get_loc('fib_take_profit1')] = extensions[0.618]
                result.iloc[i, result.columns.get_loc('fib_take_profit2')] = extensions[1.0]
                result.iloc[i, result.columns.get_loc('fib_take_profit3')] = extensions[1.618]
    
    return result

def plot_fibonacci_levels(df, lookback=50):
    """
    Plot price chart with Fibonacci levels
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Fibonacci signals
    lookback : int
        Number of bars to display
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Find rows with Fibonacci signals
    fib_rows = df[df['fib_retracement']]
    
    if len(fib_rows) == 0:
        print("No Fibonacci retracements found to plot")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for each signal (most recent one)
    latest_fib = fib_rows.iloc[-1]
    index_pos = df.index.get_loc(latest_fib.name)
    
    # Get data to plot
    plot_data = df.iloc[max(0, index_pos - lookback):min(len(df), index_pos + 20)]
    
    # Plot candlesticks
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        # Plot candle body
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([i, i], [row['open'], row['close']], color=color, linewidth=4)
        # Plot wicks
        ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
    
    # Plot Fibonacci levels
    start_price = latest_fib['fib_start_price']
    end_price = latest_fib['fib_end_price']
    
    if start_price is not None and end_price is not None:
        fib_levels = calculate_fibonacci_levels(start_price, end_price)
        
        # Plot horizontal lines for each level
        for level, price in fib_levels.items():
            if 0 < level < 1:  # Skip 0 and 1 levels
                ax.axhline(y=price, color='blue', linestyle='--', alpha=0.7,
                          label=f'Fib {level:.3f}: {price:.2f}')
                ax.text(len(plot_data) + 1, price, f'{level:.3f}', color='blue')
        
        # Highlight touched level
        level_touched = latest_fib['fib_level_touched']
        if level_touched is not None and level_touched in fib_levels:
            price = fib_levels[level_touched]
            ax.axhline(y=price, color='red', linestyle='-', alpha=0.9,
                      label=f'Touched: {level_touched:.3f}')
    
    # Add signal and potential targets if present
    signal = latest_fib['fib_signal']
    if signal:
        signal_text = f"Signal: {signal.upper()}"
        
        # Add stop loss if present
        if latest_fib['fib_stop_loss'] is not None:
            sl_price = latest_fib['fib_stop_loss']
            ax.axhline(y=sl_price, color='red', linestyle='-', linewidth=2, alpha=0.8)
            ax.text(len(plot_data) + 1, sl_price, f'Stop: {sl_price:.2f}', color='red')
            
        # Add take profit targets if present
        for i, tp_col in enumerate(['fib_take_profit1', 'fib_take_profit2', 'fib_take_profit3']):
            if latest_fib[tp_col] is not None:
                tp_price = latest_fib[tp_col]
                ax.axhline(y=tp_price, color='green', linestyle='-', linewidth=1, alpha=0.6 + i*0.1)
                ax.text(len(plot_data) + 1, tp_price, f'TP{i+1}: {tp_price:.2f}', color='green')
    
    # Set labels and title
    ax.set_title(f"Fibonacci Retracement Analysis - {latest_fib.name}")
    ax.set_ylabel("Price")
    ax.set_xticks(range(0, len(plot_data), 5))
    ax.set_xticklabels([plot_data.index[i].strftime('%m-%d %H:%M') if i < len(plot_data.index) else '' 
                        for i in range(0, len(plot_data), 5)])
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig

def generate_fibonacci_signals(df, lookback=50, swing_lookback=10, trend_window=20, atr_factor=1.5):
    """
    Generate complete Fibonacci-based trading signals
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC price data
    lookback : int
        Number of bars to look back for analysis
    swing_lookback : int
        Lookback for swing point detection
    trend_window : int
        Window for trend detection
    atr_factor : float
        Multiplier for ATR-based stop loss buffer
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Fibonacci signal columns
    """
    # Identify retracements
    result = identify_fibonacci_retracements(df, lookback, swing_lookback, trend_window)
    
    # Set targets
    result = set_fibonacci_targets(result, atr_factor)
    
    return result