# test_different_config.py
import os
from xauusd_bot import XAUUSDCentScalpingBot

# Completely different parameters
TEST_PARAMS = {
    "initial_capital": 100000,
    "risk_percent": 0.25,
    "use_fixed_lot_size": True,  # Force fixed lot size
    "fixed_lot_size": 0.01,
    "fast_ma_period": 20,        # Very different from default
    "slow_ma_period": 200,       # Very different from default
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_upper_level": 90,     # Extreme values
    "stoch_lower_level": 10,
    "take_profit_points": 150,   # Much larger take profit
    "stop_loss_points": 50,      # Much larger stop loss
    "trading_hours": {"start": 10, "end": 14},  # Different hours
    "max_daily_loss": -2.0,      # Very tight daily loss limit
    "use_adx_filter": False,     # Turn off ADX filter
    "use_rsi_filter": False      # Turn off RSI filter
}

def main():
    print("\n=== TESTING WITH COMPLETELY DIFFERENT PARAMETERS ===")
    for key, value in TEST_PARAMS.items():
        print(f"{key}: {value}")
    print("================================================\n")
    
    bot = XAUUSDCentScalpingBot(**TEST_PARAMS)
    
    print("Running backtest with radically different parameters...")
    performance = bot.backtest('xau.csv', use_timeframe='5T')
    
    if performance:
        print("\n=== Performance Summary ===")
        key_metrics = ["total_trades", "win_rate", "profit_factor", "total_profit_usd"]
        for metric in key_metrics:
            value = performance.get(metric, "N/A")
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()