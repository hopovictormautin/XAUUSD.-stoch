"""
Configuration settings for the XAUUSD Trading Bot.

This module contains default parameters and configuration settings.
"""

# Default bot parameters
# Updated DEFAULT_PARAMS for config.py
DEFAULT_PARAMS = {
    # Account/Risk parameters
    "initial_capital": 100000,    # 1000 USD in cents
    "risk_percent": 0.5,          # Reduced from 1.0% to 0.5% per trade
    "use_fixed_lot_size": False,  # Dynamic position sizing
    "fixed_lot_size": 0.01,       # Fixed lot size if enabled
    "max_lot_size": 0.03,         # Reduced from 0.05
    "min_lot_size": 0.01,         # Minimum position
    "recovery_factor": 0.25,      # More aggressive risk reduction (from 0.5)

        # Add these parameters to your configuration
    "market_regime_lookback": 20,  # Periods to look back for regime detection
    "high_volatility_risk_factor": 0.7,  # Reduce risk to 70% in high volatility
    "low_volatility_risk_factor": 1.2,  # Increase risk to 120% in low volatility
    "r_r_ratio": 2.0,  # Your target reward-to-risk ratio
    
    # Trend and entry parameters
    "fast_ma_period": 5,          # Keep original
    "slow_ma_period": 100,        # Keep original
    "stoch_k_period": 5,          # Keep original
    "stoch_d_period": 2,          # Keep original
    "stoch_slowing": 3,           # Keep original
    "stoch_upper_level": 92,      # More extreme (from 80)
    "stoch_lower_level": 22,      # More extreme (from 20)
    
    # Filters
    "use_adx_filter": True,
    "adx_period": 14,
    "adx_threshold": 20,          # Reduced from 25
    "use_rsi_filter": True,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    
    # Exit parameters
    "take_profit_points": 30,     # Keep original
    "stop_loss_points": 20,       # Keep original
    #"r_r_ratio": 2.0,             # Increased from 1.5
    
    # Dynamic exit improvements
    "use_dynamic_exits": True,
    "trailing_stop": True,
    "trailing_distance": 25,
    "use_breakeven_stop": True,
    "breakeven_trigger_ratio": 0.4,
    "max_trade_duration_minutes": 60,  # Reduced from 480 to 4 hours
    
    # Partial profit taking
    "use_partial_take_profit": True,
    "partial_tp_ratio": 0.5,
    "partial_tp_distance": 0.5,
    
    # Session and risk protection
    "trading_hours": {"start": 8, "end": 20},  # Adjusted to avoid volatile opens
    "max_daily_loss": -2.5,                    # Tighter daily loss limit (from -5.0)
    "consecutive_losses_limit": 2,             # React faster to losses (from 3)
    
    # Volatility filters
    "use_atr_filter": True,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "avg_spread_points": 25,

    # volume
    "use_volume_filter": True,
    "volume_lookback": 20,          # Periods to calculate average volume
    "volume_threshold": 0.7,        # Minimum relative volume (70% of average)
    "volume_spike_threshold": 2.5  # Maximum relative volume (250% of average)

        
}
# Default parameter ranges for optimization
DEFAULT_PARAM_RANGES = {
    "fast_ma_period": [5, 8, 10, 12, 15],
    "slow_ma_period": [50, 100, 150, 200],
    "stoch_k_period": [5, 7, 9, 14],
    "stoch_upper_level": [70, 75, 80, 85],
    "stoch_lower_level": [15, 20, 25, 30],
    "take_profit_points": [50, 75, 100],
    "stop_loss_points": [30, 45, 60],
    "use_adx_filter": [True, False],
    "trailing_stop": [True, False],
    "adx_threshold": [20, 25, 30],
    "r_r_ratio": [1.0, 1.5, 2.0]
}

# File paths
DATA_PATH = "C:/Users/V00426/OneDrive - Uniper SE/Desktop/BOT. 20250402/data/xau.csv"
RESULTS_PATH = "results/"
REPORTS_PATH = "results/reports/"
OPTIMIZATION_PATH = "results/optimization/"
CHARTS_PATH = "results/charts/"