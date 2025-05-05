"""
Technical indicators calculation module for XAUUSD Bot.

This module provides vectorized implementation of technical indicators
used by the trading bot for improved efficiency.
"""

import pandas as pd
import numpy as np

def calculate_indicators(df, bot):
    """
    Calculate all technical indicators needed for the strategy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Price data with OHLCV columns
    bot : XAUUSDCentScalpingBot
        Bot instance with indicator parameters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added indicator columns
    """
    # Moving averages - using pandas rolling for efficiency
    df["fast_ma"] = df["close"].rolling(bot.fast_ma_period).mean()
    df["slow_ma"] = df["close"].rolling(bot.slow_ma_period).mean()
    
    # Higher timeframe trend (optional)
    # df["higher_tf_ma"] = df["close"].rolling(100).mean()
    
    # Stochastic - full vectorized calculation
    df["lowest_low"] = df["low"].rolling(bot.stoch_k_period).min()
    df["highest_high"] = df["high"].rolling(bot.stoch_k_period).max()
    
    # Avoid division by zero
    range_diff = df["highest_high"] - df["lowest_low"]
    range_diff = range_diff.replace(0, np.nan)  # Replace zeros with NaN
    
    df["stoch_k"] = 100 * (df["close"] - df["lowest_low"]) / range_diff
    
    if bot.stoch_slowing > 1:
        df["stoch_k"] = df["stoch_k"].rolling(bot.stoch_slowing).mean()
    df["stoch_d"] = df["stoch_k"].rolling(bot.stoch_d_period).mean()

    # RSI - vectorized
    if bot.use_rsi_filter:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(bot.rsi_period).mean()
        avg_loss = loss.rolling(bot.rsi_period).mean()
        
        # Avoid division by zero
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        df["rsi"] = 100 - (100 / (1 + rs))

    # ATR calculation - vectorized
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(bot.atr_period).mean()
    
    if bot.use_atr_filter:
        df["atr_threshold"] = df["atr"] * bot.atr_multiplier
    
    # ADX for trend strength filtering
    if bot.use_adx_filter:
        calculate_adx(df, bot.adx_period)

    # Clean up dataframe
    df.dropna(inplace=True)
    return df

def calculate_adx(df, period=14):
    """
    Calculate ADX (Average Directional Index) for trend strength.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Price data with OHLCV columns
    period : int
        ADX calculation period
        
    Adds columns:
    -------------
    adx : Average Directional Index
    plus_di : +DI line
    minus_di : -DI line
    """
    # True Range
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift())
    tr3 = abs(df["low"] - df["close"].shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Plus and Minus Directional Movement
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff() * -1
    
    # Only keep positive values where +DM > -DM or -DM > +DM
    plus_dm = np.where((plus_dm > 0) & (plus_dm > minus_dm), plus_dm, 0)
    minus_dm = np.where((minus_dm > 0) & (minus_dm > plus_dm), minus_dm, 0)
    
    # Smoothed +DM and -DM
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    
    # Directional Movement Index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero 
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Average Directional Index
    df["adx"] = pd.Series(dx).rolling(period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    
    return df