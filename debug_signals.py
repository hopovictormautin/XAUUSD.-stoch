# debug_signals.py
import os
import pandas as pd
from xauusd_bot import XAUUSDCentScalpingBot
from config import DEFAULT_PARAMS

def main():
    """Debug the signal generation of the XAUUSD bot"""
    # Create bot with default parameters
    bot = XAUUSDCentScalpingBot(**DEFAULT_PARAMS)
    
    # Load data
    data = bot.load_data('xau.csv')
    if data is None:
        print("Could not load data!")
        return
    
    # Resample to 5-minute timeframe
    data = bot.resample_data(data, '5T')
    
    # Prepare data with indicators
    df = bot.prepare_data(data.copy())
    
    # Limit to 100 rows to avoid too much output
    print(f"Total rows after preparation: {len(df)}")
    sample_size = min(100, len(df))
    sample_df = df.iloc[:sample_size]
    
    print("\n=== CHECKING SIGNAL GENERATION ON SAMPLE DATA ===")
    print(f"Using first {sample_size} rows (after indicator calculations)\n")
    
    # Check signal counts
    signal_counts = {
        'buy_signals': 0,
        'sell_signals': 0,
        'filtered_by_trading_hours': 0,
        'filtered_by_atr': 0,
        'filtered_by_adx': 0,
        'filtered_by_rsi': 0,
        'stoch_conditions_met': 0,
        'ma_conditions_met': 0
    }
    
    # Loop through sample data
    for idx, row in sample_df.iterrows():
        print(f"\nChecking row at {idx}")
        
        # Trading hours filter
        hour = idx.hour
        if hour < DEFAULT_PARAMS['trading_hours']['start'] or hour >= DEFAULT_PARAMS['trading_hours']['end']:
            print(f"  Outside trading hours ({DEFAULT_PARAMS['trading_hours']['start']}-{DEFAULT_PARAMS['trading_hours']['end']})")
            signal_counts['filtered_by_trading_hours'] += 1
            continue
            
        # MA condition
        ma_bullish = row['fast_ma'] > row['slow_ma']
        ma_bearish = row['fast_ma'] < row['slow_ma']
        print(f"  MA condition: Fast MA ({row['fast_ma']:.2f}) {'>' if ma_bullish else '<'} Slow MA ({row['slow_ma']:.2f})")
        if ma_bullish or ma_bearish:
            signal_counts['ma_conditions_met'] += 1
            
        # Stochastic condition
        stoch_oversold = row['stoch_k'] < DEFAULT_PARAMS['stoch_lower_level']
        stoch_overbought = row['stoch_k'] > DEFAULT_PARAMS['stoch_upper_level']
        k_above_d = row['stoch_k'] > row['stoch_d']
        k_below_d = row['stoch_k'] < row['stoch_d']
        
        print(f"  Stoch K: {row['stoch_k']:.2f}, D: {row['stoch_d']:.2f}")
        print(f"  Oversold: {stoch_oversold}, Overbought: {stoch_overbought}")
        print(f"  K>D: {k_above_d}, K<D: {k_below_d}")
        
        if (stoch_oversold and k_above_d) or (stoch_overbought and k_below_d):
            signal_counts['stoch_conditions_met'] += 1
            
        # ATR filter if enabled
        if DEFAULT_PARAMS['use_atr_filter']:
            current_volatility = abs(row['high'] - row['low'])
            if 'atr_threshold' in row and current_volatility > row['atr_threshold']:
                print(f"  Filtered by ATR: current volatility {current_volatility:.5f} > threshold {row['atr_threshold']:.5f}")
                signal_counts['filtered_by_atr'] += 1
                continue
                
        # ADX filter if enabled
        if DEFAULT_PARAMS['use_adx_filter'] and 'adx' in row:
            if row['adx'] > DEFAULT_PARAMS['adx_threshold']:
                print(f"  Filtered by ADX: {row['adx']:.2f} > {DEFAULT_PARAMS['adx_threshold']}")
                signal_counts['filtered_by_adx'] += 1
                continue
                
        # Check buy signal
        buy_condition = ma_bullish and stoch_oversold and k_above_d
        if buy_condition:
            if DEFAULT_PARAMS['use_rsi_filter'] and 'rsi' in row:
                if row['rsi'] <= DEFAULT_PARAMS['rsi_oversold']:
                    print(f"  Filtered by RSI: {row['rsi']:.2f} <= {DEFAULT_PARAMS['rsi_oversold']} (buy)")
                    signal_counts['filtered_by_rsi'] += 1
                    continue
            print(f"  BUY SIGNAL at {idx}")
            signal_counts['buy_signals'] += 1
            
        # Check sell signal
        sell_condition = ma_bearish and stoch_overbought and k_below_d
        if sell_condition:
            if DEFAULT_PARAMS['use_rsi_filter'] and 'rsi' in row:
                if row['rsi'] >= DEFAULT_PARAMS['rsi_overbought']:
                    print(f"  Filtered by RSI: {row['rsi']:.2f} >= {DEFAULT_PARAMS['rsi_overbought']} (sell)")
                    signal_counts['filtered_by_rsi'] += 1
                    continue
            print(f"  SELL SIGNAL at {idx}")
            signal_counts['sell_signals'] += 1
    
    # Print summary
    print("\n=== SIGNAL GENERATION SUMMARY ===")
    print(f"Sample size: {sample_size} bars")
    print(f"MA conditions met: {signal_counts['ma_conditions_met']}")
    print(f"Stochastic conditions met: {signal_counts['stoch_conditions_met']}")
    print(f"Filtered by trading hours: {signal_counts['filtered_by_trading_hours']}")
    print(f"Filtered by ATR: {signal_counts['filtered_by_atr']}")
    print(f"Filtered by ADX: {signal_counts['filtered_by_adx']}")
    print(f"Filtered by RSI: {signal_counts['filtered_by_rsi']}")
    print(f"Buy signals generated: {signal_counts['buy_signals']}")
    print(f"Sell signals generated: {signal_counts['sell_signals']}")
    print(f"Total signals: {signal_counts['buy_signals'] + signal_counts['sell_signals']}")
    
    # Suggestion based on results
    if signal_counts['buy_signals'] + signal_counts['sell_signals'] == 0:
        print("\nNo signals were generated in the sample. Possible reasons:")
        
        if signal_counts['ma_conditions_met'] == 0:
            print("- MA conditions are never met. Consider adjusting fast/slow MA periods.")
        
        if signal_counts['stoch_conditions_met'] == 0:
            print("- Stochastic conditions are never met. Consider adjusting stoch levels.")
        
        if signal_counts['filtered_by_trading_hours'] > sample_size * 0.5:
            print("- Most bars are outside trading hours. Consider adjusting trading hours.")
            
        if signal_counts['filtered_by_atr'] > 0:
            print("- ATR filter is removing potential signals. Consider adjusting atr_multiplier.")
            
        if signal_counts['filtered_by_adx'] > 0:
            print("- ADX filter is removing potential signals. Consider increasing adx_threshold.")
            
        if signal_counts['filtered_by_rsi'] > 0:
            print("- RSI filter is removing potential signals. Consider adjusting rsi thresholds.")
            
        print("\nSuggested parameter adjustments:")
        print("1. Widen Stochastic levels: stoch_upper_level=90, stoch_lower_level=10")
        print("2. Increase ADX threshold to 30 or disable ADX filter")
        print("3. Adjust RSI levels: rsi_overbought=80, rsi_oversold=20")
        print("4. Make sure trading hours cover active market periods")

if __name__ == "__main__":
    main()