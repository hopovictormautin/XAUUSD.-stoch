"""
XAUUSD Cent Account Scalping Bot
Core trading bot class implementation with backtest capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import warnings
import concurrent.futures
import time
import os
from utils.indicators import calculate_indicators
from utils.plotting import plot_equity_curve, plot_monthly_performance, plot_trade_analysis
from utils.reporting import generate_performance_report

# Suppress pandas DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

class XAUUSDCentScalpingBot:
    def __init__(
        self,
        *,
        initial_capital=100_000,           # 1000 USD represented as 100 000 ¢
        risk_percent=1.0,                  # Risk % per trade
        take_profit_points=30,             # 50 points (5 pips)
        stop_loss_points=20,               # 35 points (3.5 pips)
        fast_ma_period=5,
        slow_ma_period=21,
        stoch_k_period=5,
        stoch_d_period=2,
        stoch_slowing=3,
        stoch_upper_level=75,
        stoch_lower_level=25,
        avg_spread_points=25,
        use_fixed_lot_size=False,
        fixed_lot_size=0.01,
        max_lot_size=0.05,
        min_lot_size=0.01,
        trailing_stop=False,
        trailing_distance=20,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        use_rsi_filter=True,
        use_atr_filter=True,
        atr_period=14,
        atr_multiplier=1.5,
        adx_period=14,               # Added ADX period for trend strength
        adx_threshold=25,            # Added ADX threshold for filtering
        use_adx_filter=False,        # Toggle for ADX trend filter
        trading_hours=None,          # dict {'start': int, 'end': int}
        max_daily_loss=-5.0,         # % of account
        recovery_factor=0.5,         # size reduction after losses
        consecutive_losses_limit=3,
        max_trade_duration_minutes=15,  # Maximum time to hold a trade (8 hours)
        use_partial_take_profit=False,   # Enable partial profit taking
        partial_tp_ratio=0.5,            # Percentage of position to close at first target
        partial_tp_distance=0.5,         # First TP distance as ratio of full TP
        r_r_ratio=2,                   # Reward to risk ratio for dynamic TP setting
        use_dynamic_exits=False,         # Use ATR-based stops and targets
        use_breakeven_stop=False,        # Move stop to breakeven after certain profit
        breakeven_trigger_ratio=0.4,      # Move to breakeven after reaching 0.5× risk
        market_regime_lookback=20,
        high_volatility_risk_factor=0.7,
        low_volatility_risk_factor=1.2,
        use_volume_filter: False,
        volume_lookback: 20,          # Periods to calculate average volume
        volume_threshold: 0.7,        # Minimum relative volume (70% of average)
        volume_spike_threshold: 2.5  # Maximum relative volume (250% of average)
    ):
        # Session hours (defaults to 08-16)
        self.trading_hours = trading_hours or {"start": 8, "end": 16}

        # ── account │ risk ──────────────────────────────────────────
        self.initial_capital = initial_capital      # cents
        self.current_balance = initial_capital      # cents
        self.risk_percent = risk_percent
        self.max_daily_loss = max_daily_loss
        self.use_fixed_lot_size = use_fixed_lot_size
        self.fixed_lot_size = fixed_lot_size
        self.min_lot_size = min_lot_size
        self.max_lot_size = max_lot_size
        self.recovery_factor = recovery_factor
        self.consecutive_losses_limit = consecutive_losses_limit

        # ── trade params ────────────────────────────────────────────
        self.take_profit_points = take_profit_points
        self.stop_loss_points = stop_loss_points
        self.avg_spread_points = avg_spread_points
        self.trailing_stop = trailing_stop
        self.trailing_distance = trailing_distance
        self.max_trade_duration_minutes = max_trade_duration_minutes
        self.use_partial_take_profit = use_partial_take_profit
        self.partial_tp_ratio = partial_tp_ratio
        self.partial_tp_distance = partial_tp_distance
        self.r_r_ratio = r_r_ratio
        self.use_dynamic_exits = use_dynamic_exits
        self.use_breakeven_stop = use_breakeven_stop
        self.breakeven_trigger_ratio = breakeven_trigger_ratio

        # ── strategy params ─────────────────────────────────────────
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_slowing = stoch_slowing
        self.stoch_upper_level = stoch_upper_level
        self.stoch_lower_level = stoch_lower_level
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_rsi_filter = use_rsi_filter
        self.use_atr_filter = use_atr_filter
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_adx_filter = use_adx_filter

        # Add these lines to store the new parameters
        self.use_volume_filter = use_volume_filter
        self.volume_lookback = volume_lookback
        self.volume_threshold = volume_threshold
        self.volume_spike_threshold = volume_spike_threshold
    

        # ── runtime state ───────────────────────────────────────────
        self.position = None            # 'buy', 'sell', or None
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.trailing_stop_price = 0
        self.position_size = 0
        self.consecutive_losses = 0
        self.daily_profit_loss = 0
        self.last_trade_date = None
        self.entry_time = None
        self.partial_exit_done = False
        self.original_position_size = 0

        # ── performance tracking ────────────────────────────────────
        self.trades = []
        self.equity_curve = []
        self.daily_results = {}
        self.trade_details = {}

        # ── debug counters ──────────────────────────────────────────
        self.signal_counts = {
            "ma_bullish": 0,
            "ma_bearish": 0,
            "stoch_oversold": 0,
            "stoch_overbought": 0, 
            "k_above_d": 0,
            "k_below_d": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "rsi_buy_confirmed": 0,
            "rsi_sell_confirmed": 0,
            "atr_filtered_out": 0,
            "adx_filtered_out": 0,
            "time_exit": 0,
            "partial_tp_hit": 0,
            "breakeven_activated": 0
        }

    # ─────────────────────── data helpers ──────────────────────────
    def load_data(self, csv_file: str) -> Optional[pd.DataFrame]:
        """
        Load & prepare CSV (semicolon-delimited, cent-account price format).

        Parameters
        ----------
        csv_file : str
            Full path (or relative path) to the CSV file.

        Returns
        -------
        Optional[pandas.DataFrame]
            A prepared DataFrame, or None if an error occurs.
        """
        try:
            # read the file that was passed in, not a hard-coded path
            df = pd.read_csv(csv_file, sep=";", skiprows=1)

            # clean column names
            df.columns = [c.replace('"', '') for c in df.columns]

            # rename to standard names if present
            mapping = {
                "Timestamp": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

            # strip stray quotes from string columns
            for col in df.select_dtypes(include="object"):
                df[col] = df[col].str.replace('"', '')

            # parse datetime and set index
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            # force numeric types
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # synthesize bid/ask and spread
            spread_val = self.avg_spread_points * 0.001
            df["ask"] = df["close"] + spread_val / 2
            df["bid"] = df["close"] - spread_val / 2
            df["spread_points"] = (df["ask"] - df["bid"]) * 1000

            print(f"Data loaded: {len(df)} rows ({df.index.min()} → {df.index.max()})")
            return df

        except Exception as exc:
            print(f"Error loading data: {exc}")
            return None
    # Modified backtest method to fix the Timestamp comparison error
    def backtest(self, csv_file, *, use_timeframe=None):
        """Run backtest on historical data with improved efficiency"""
        # Load and prepare data 
        start_time = datetime.now()
        print(f"Starting backtest at {start_time.strftime('%H:%M:%S')}")
        
        data = self.load_data(csv_file)
        if data is None:
            return None

        if use_timeframe:
            data = self.resample_data(data, use_timeframe)

        df = self.prepare_data(data.copy())
        
        # Reset performance tracking
        self.trades.clear()
        self.equity_curve = [self.initial_capital]
        self.current_balance = self.initial_capital
        self.position = None
        self.consecutive_losses = 0
        self.daily_profit_loss = 0
        self.last_trade_date = None
        current_trade = None

        # Track market regimes for analysis
        regime_changes = []
        current_regime = "unknown"
        
        # Counter to track data rows processed
        row_count = 0

        # Run the backtest using iterrows
        for idx, row in df.iterrows():
            row_count += 1
            
            # Update equity curve
            self.equity_curve.append(self.current_balance)

            # Calculate current market regime (after we have enough data)
            if row_count > 20:  # Using row counter instead of timestamp comparison
                lookback_data = df.iloc[max(0, row_count-20):row_count]
                new_regime = self.detect_market_regime(lookback_data)
                if new_regime != current_regime:
                    regime_changes.append((idx, current_regime, new_regime))
                    current_regime = new_regime
                    print(f"Market regime changed at {idx}: {current_regime}")

            # Check for exit if in a position
            if self.position is not None:
                exit_now, profit, reason = self.check_exit_conditions(row)
                if exit_now:
                    trade_res = self.close_position(row, profit, reason)
                    current_trade.update(trade_res)
                    self.trades.append(current_trade)
                    current_trade = None
            
            # Check for entry if not in a position
            if self.position is None:
                sig = self.check_entry_signals(row)
                if sig:
                    # For market regime, use a slice of recent data
                    lookback_data = df.iloc[max(0, row_count-20):row_count]
                    current_trade = self.open_position(sig, row, lookback_data)
                    self.trade_details[idx] = {
                        "type": sig,
                        "entry_price": self.entry_price,
                        "stop_loss": self.stop_loss_price,
                        "take_profit": self.take_profit_price,
                        "market_regime": current_trade.get("market_regime", "normal_volatility")
                    }
        
        # Continue with the rest of the function (reset performance tracking, etc.)
    def resample_data(self, df: pd.DataFrame, timeframe="5T") -> pd.DataFrame:
        """
        Resample OHLCV to a new timeframe (e.g. '5T', '1H', '4H', 'D').
        """
        print(f"Resampling to {timeframe}")
        res = (
            df.resample(timeframe)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        spread_val = self.avg_spread_points * 0.001
        res["ask"] = res["close"] + spread_val / 2
        res["bid"] = res["close"] - spread_val / 2
        return res

    # def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Compute indicators using the utility function.
    #     """
    #     return calculate_indicators(df, self)
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators used by the strategy."""
        print(f"Starting data preparation with {len(df)} rows")
        
        # Moving averages
        print("Calculating moving averages...")
        df["fast_ma"] = df["close"].rolling(self.fast_ma_period).mean()
        df["slow_ma"] = df["close"].rolling(self.slow_ma_period).mean()
        
        # Stochastic
        print("Calculating Stochastic...")
        df["lowest_low"] = df["low"].rolling(self.stoch_k_period).min()
        df["highest_high"] = df["high"].rolling(self.stoch_k_period).max()
        
        # Avoid division by zero
        range_diff = df["highest_high"] - df["lowest_low"]
        range_diff = range_diff.replace(0, np.nan)  # Replace zeros with NaN
        
        df["stoch_k"] = 100 * (df["close"] - df["lowest_low"]) / range_diff
        
        if self.stoch_slowing > 1:
            df["stoch_k"] = df["stoch_k"].rolling(self.stoch_slowing).mean()
        df["stoch_d"] = df["stoch_k"].rolling(self.stoch_d_period).mean()

        # RSI
        if self.use_rsi_filter:
            print("Calculating RSI...")
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(self.rsi_period).mean()
            avg_loss = loss.rolling(self.rsi_period).mean()
            
            # Avoid division by zero
            rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
            df["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        if self.use_atr_filter:
            print("Calculating ATR...")
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df["atr"] = true_range.rolling(self.atr_period).mean()
            df["atr_threshold"] = df["atr"] * self.atr_multiplier
        
        # ADX for trend strength filtering
        if self.use_adx_filter:
            print("Calculating ADX...")
            # Add ADX calculation here
            # ... [existing ADX calculation code] ...

        # Count rows before and after dropping NaN values
        print(f"Row count before dropping NaN: {len(df)}")
        print(f"NaN values in dataframe: {df.isna().sum().sum()}")
        
        # Show rows with NaN values
        if df.isna().any().any():
            print("Sample rows with NaN values:")
            print(df[df.isna().any(axis=1)].head(3))
        
        # Instead of dropping all NaN values, fill them forward
        df = df.fillna(method='ffill')
        
        # Only drop rows at the beginning that would have NaNs
        # due to indicator calculation windows
        max_window = max(
            self.slow_ma_period,
            self.stoch_k_period + self.stoch_d_period,
            self.rsi_period if self.use_rsi_filter else 0,
            self.atr_period if self.use_atr_filter else 0,
            self.adx_period if self.use_adx_filter else 0
        )
        
        # Drop only the beginning rows where indicators cannot be calculated
        if len(df) > max_window:
            df = df.iloc[max_window:]
        
        print(f"Row count after preparation: {len(df)}")

            # Add volume analysis
        if 'volume' in df.columns and self.use_volume_filter:
            # Calculate average volume
            lookback = getattr(self, 'volume_lookback', 20)
            df['avg_volume'] = df['volume'].rolling(lookback).mean()
            
            # Calculate relative volume
            df['relative_volume'] = df['volume'] / df['avg_volume']
            
            # Mark low volume periods
            volume_threshold = getattr(self, 'volume_threshold', 0.7)
            df['low_volume'] = df['relative_volume'] < volume_threshold
            
            # Mark high volume spikes (potentially dangerous for scalping)
            spike_threshold = getattr(self, 'volume_spike_threshold', 2.5)
            df['volume_spike'] = df['relative_volume'] > spike_threshold
            
            print(f"Added volume filters: low < {volume_threshold}, spike > {spike_threshold}")
        return df
    # ────────────────────── utility helpers ────────────────────────
    @staticmethod
    def _trade_date(ts):
        return ts.date() if isinstance(ts, pd.Timestamp) else ts.date()

    def reset_daily_stats(self, current_date):
        """Reset daily stats and check daily loss limit"""
        if self.last_trade_date is None or current_date != self.last_trade_date:
            if self.last_trade_date is not None:
                self.daily_results[self.last_trade_date] = self.daily_profit_loss
            self.daily_profit_loss = 0
            print(f"── new trading day {current_date} ──")
        self.last_trade_date = current_date

    def calculate_lot_size(self, stop_loss_points):
        """Calculate position size based on risk management rules"""
        if self.use_fixed_lot_size:
            return self.fixed_lot_size

        usd_balance = self.current_balance / 100  # Convert cents to USD
        risk_usd = usd_balance * self.risk_percent / 100  # Amount to risk in USD

        # IMPORTANT: Use the absolute value of the balance when it's negative
        # This prevents trying to risk a percentage of a negative number
        if usd_balance < 0:
            usd_balance = abs(usd_balance)
            risk_usd = min(5.0, usd_balance * self.risk_percent / 100)  # Cap risk when in drawdown

        # Reduce position size after consecutive losses
        if self.consecutive_losses >= self.consecutive_losses_limit:
            risk_usd *= self.recovery_factor
            print(f"Reducing risk after {self.consecutive_losses} consecutive losses")

        # FIXED: Correct pip value for XAUUSD
        # For XAUUSD, 1 pip (0.01 move) = $0.01 per 0.01 lot
        value_per_point_per_01lot = 0.01  # $ per point per 0.01 lot
        
        # Calculate lot size
        # Points to pips conversion - in gold, 1 point = 0.001, 1 pip = 0.01
        stop_loss_pips = stop_loss_points / 10
        
        lots = risk_usd / (stop_loss_pips * value_per_point_per_01lot)
        lots = math.floor(lots * 100) / 100  # Round down to nearest 0.01
        lots = max(self.min_lot_size, min(lots, self.max_lot_size))

        # Minimum balance safety check
        if self.current_balance < 5_000:  # < $50
            lots = self.min_lot_size
            print("Low balance: using minimum lot size")
            
        return lots
    # ────────────────────── signal generation ──────────────────────
    def check_entry_signals(self, row):
        """Check if there's an entry signal based on current conditions"""
        # Extract hour from timestamp
        hour = row.name.hour if isinstance(row.name, pd.Timestamp) else row.name.hour
        
        # Debug hourly check
        print(f"Checking signals at {row.name}, hour: {hour}")
        
        # Check if we're in trading hours
        if hour < self.trading_hours['start'] or hour >= self.trading_hours['end']:
            print(f"Outside trading hours ({self.trading_hours['start']}-{self.trading_hours['end']}), skipping")
            return None
        
        # Get current date for daily tracking
        current_date = self.get_trade_date(row.name)
        self.reset_daily_stats(current_date)
        
        # Check if we hit the daily loss limit
        daily_loss_limit_cents = self.initial_capital * self.max_daily_loss / 100
        if self.daily_profit_loss <= daily_loss_limit_cents:
            print(f"Daily loss limit reached: {self.daily_profit_loss:.2f} cents. No more trading today.")
            return None
        
        # Debug indicator values for signal generation
        print(f"Indicators: Fast MA: {row['fast_ma']:.2f}, Slow MA: {row['slow_ma']:.2f}, K: {row['stoch_k']:.2f}, D: {row['stoch_d']:.2f}")
        if 'rsi' in row:
            print(f"RSI: {row['rsi']:.2f}")
        if 'adx' in row:
            print(f"ADX: {row['adx']:.2f}")
        
        # Track indicator conditions for debugging
        ma_bullish = row['fast_ma'] > row['slow_ma']
        ma_bearish = row['fast_ma'] < row['slow_ma']
        stoch_oversold = row['stoch_k'] < self.stoch_lower_level
        stoch_overbought = row['stoch_k'] > self.stoch_upper_level
        k_above_d = row['stoch_k'] > row['stoch_d']
        k_below_d = row['stoch_d'] > row['stoch_k']
        
        # Debug signal conditions
        print(f"MA Bullish: {ma_bullish}, MA Bearish: {ma_bearish}")
        print(f"Stoch Oversold: {stoch_oversold}, Stoch Overbought: {stoch_overbought}")
        print(f"K > D: {k_above_d}, K < D: {k_below_d}")
        
        # Update signal counters
        if ma_bullish: self.signal_counts['ma_bullish'] += 1
        if ma_bearish: self.signal_counts['ma_bearish'] += 1
        if stoch_oversold: self.signal_counts['stoch_oversold'] += 1
        if stoch_overbought: self.signal_counts['stoch_overbought'] += 1
        if k_above_d: self.signal_counts['k_above_d'] += 1
        if k_below_d: self.signal_counts['k_below_d'] += 1
        
        # Check for volatility filtering with ATR
        if self.use_atr_filter and 'atr' in row and 'atr_threshold' in row:
            current_volatility = abs(row['high'] - row['low']) 
            print(f"Volatility check: Current: {current_volatility:.5f}, Threshold: {row['atr_threshold']:.5f}")
            if current_volatility > row['atr_threshold']:
                self.signal_counts['atr_filtered_out'] += 1
                print("Filtered out by ATR (too volatile)")
                return None
        
        # ADX filter for trend strength
        if self.use_adx_filter and 'adx' in row:
            # For mean-reversion trades, we want low ADX (range-bound market)
            print(f"ADX check: Current: {row['adx']:.2f}, Threshold: {self.adx_threshold}")
            if row['adx'] > self.adx_threshold:
                self.signal_counts['adx_filtered_out'] += 1
                print("Filtered out by ADX (trending market)")
                return None
        
        # BUY signal conditions
        buy_signal_condition = ma_bullish and stoch_oversold and k_above_d
        sell_signal_condition = ma_bearish and stoch_overbought and k_below_d
        
        print(f"Buy signal check: {buy_signal_condition}, Sell signal check: {sell_signal_condition}")
        
        # Check for BUY signal
        if buy_signal_condition:
            # Additional RSI filter for buy signals if enabled
            if self.use_rsi_filter and 'rsi' in row:
                rsi_condition = row['rsi'] > self.rsi_oversold
                print(f"RSI buy check: {row['rsi']:.2f} > {self.rsi_oversold} = {rsi_condition}")
                if rsi_condition:
                    self.signal_counts['rsi_buy_confirmed'] += 1
                else:
                    print("Filtered out by RSI (buy)")
                    return None
                    
            print(f"BUY SIGNAL at {row.name}: Fast MA: {row['fast_ma']:.2f}, Slow MA: {row['slow_ma']:.2f}, K: {row['stoch_k']:.2f}, D: {row['stoch_d']:.2f}")
            self.signal_counts['buy_signals'] += 1
            return 'buy'
        
        # Check for SELL signal
        elif sell_signal_condition:
            # Additional RSI filter for sell signals if enabled
            if self.use_rsi_filter and 'rsi' in row:
                rsi_condition = row['rsi'] < self.rsi_overbought
                print(f"RSI sell check: {row['rsi']:.2f} < {self.rsi_overbought} = {rsi_condition}")
                if rsi_condition:
                    self.signal_counts['rsi_sell_confirmed'] += 1
                else:
                    print("Filtered out by RSI (sell)")
                    return None
                    
            print(f"SELL SIGNAL at {row.name}: Fast MA: {row['fast_ma']:.2f}, Slow MA: {row['slow_ma']:.2f}, K: {row['stoch_k']:.2f}, D: {row['stoch_d']:.2f}")
            self.signal_counts['sell_signals'] += 1
            return 'sell'
        
        # Volume filter check
        if hasattr(self, 'use_volume_filter') and self.use_volume_filter:
            if 'low_volume' in row and row['low_volume']:
                print(f"Low volume detected: {row['volume']:.0f} vs avg {row['avg_volume']:.0f}")
                return None
                
            if 'volume_spike' in row and row['volume_spike']:
                print(f"Volume spike detected: {row['volume']:.0f} vs avg {row['avg_volume']:.0f}")
                return None
        print("No signal generated")
        return None




    def detect_market_regime(self, df, lookback=20):
        """
        Detect if market is in high or low volatility regime based on ATR.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with calculated indicators including 'atr'
        lookback : int
            Number of periods to use for average ATR calculation
            
        Returns:
        --------
        str
            "high_volatility", "low_volatility", or "normal_volatility"
        """
        if 'atr' not in df.columns or len(df) == 0:
            return "normal_volatility"
            
        current_atr = df['atr'].iloc[-1]
        
        # Calculate the lookback ATR average
        if len(df) > 1:
            # Use all available data points up to lookback
            avg_atr = df['atr'].mean()
        else:
            # If only one data point, use it as average
            avg_atr = current_atr
        
        if current_atr > avg_atr * 1.2:
            return "high_volatility"
        elif current_atr < avg_atr * 0.8:
            return "low_volatility"
        else:
            return "normal_volatility"
    # ───────────────────────── trade logic ─────────────────────────
    def open_position(self, signal, row, market_regime="normal_volatility"):
        """Open a new trade position with improved risk management"""
        # Note: Changed parameter from df to market_regime with default value
        print(f"Current market regime: {market_regime}")
        
        # Adjust risk based on market regime
        effective_risk_percent = self.risk_percent
        effective_r_r_ratio = getattr(self, 'r_r_ratio', 2.0)  # Default to 2.0 if not defined
        
        if market_regime == "high_volatility":
            # Be more conservative in volatile markets
            effective_risk_percent = self.risk_percent * 0.75
            effective_r_r_ratio = effective_r_r_ratio * 1.2
            print(f"High volatility detected: reducing risk to {effective_risk_percent:.2f}%")
        elif market_regime == "low_volatility":
            # Can be more aggressive in calm markets
            effective_r_r_ratio = effective_r_r_ratio * 0.9
            print(f"Low volatility detected: adjusting R:R to {effective_r_r_ratio:.2f}")
        
        # If using dynamic exits based on ATR
        if self.use_dynamic_exits and 'atr' in row:
            sl_distance = row['atr'] * 1.0  # 1.0 × ATR for stop loss
            tp_distance = sl_distance * effective_r_r_ratio  # Use adjusted r:r ratio
        else:
            # Use fixed points
            sl_distance = self.stop_loss_points * 0.001
            tp_distance = self.take_profit_points * 0.001

        # Calculate position size with adjusted risk
        sl_points = int(sl_distance * 1000)  # Convert to points for lot calc
        
        # Store original risk percent to restore after this trade
        original_risk_percent = self.risk_percent
        # Temporarily change risk percent for position sizing
        self.risk_percent = effective_risk_percent
        
        self.position_size = self.calculate_lot_size(sl_points)
        self.original_position_size = self.position_size
        
        # Restore original risk percent
        self.risk_percent = original_risk_percent
        
        if signal == "buy":
            self.entry_price = row["ask"]
            self.stop_loss_price = self.entry_price - sl_distance
            self.take_profit_price = self.entry_price + tp_distance
            
            # Set partial take profit if enabled
            if self.use_partial_take_profit:
                self.partial_tp_price = self.entry_price + (tp_distance * self.partial_tp_distance)
            
            if self.trailing_stop:
                self.trailing_stop_price = self.stop_loss_price
        else:  # sell
            self.entry_price = row["bid"]
            self.stop_loss_price = self.entry_price + sl_distance
            self.take_profit_price = self.entry_price - tp_distance
            
            # Set partial take profit if enabled
            if self.use_partial_take_profit:
                self.partial_tp_price = self.entry_price - (tp_distance * self.partial_tp_distance)
                
            if self.trailing_stop:
                self.trailing_stop_price = self.stop_loss_price

        self.position = signal
        self.entry_time = row.name
        self.partial_exit_done = False
        
        trade = {
            "entry_time": row.name,
            "position": signal,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "lot_size": self.position_size,
            "market_regime": market_regime  # Add market regime to trade info
        }
        
        # Add dynamic SL/TP info if used
        if self.use_dynamic_exits:
            trade["atr_at_entry"] = row.get('atr', 0)
            trade["r_r_ratio"] = effective_r_r_ratio
            
        print(
            f"Open {signal} @ {self.entry_price:.3f}  SL={self.stop_loss_price:.3f} "
            f"TP={self.take_profit_price:.3f}  lot={self.position_size:.2f}"
        )
        return trade

    def update_trailing_stop(self, row):
        """Update trailing stop based on price movement"""
        if not self.trailing_stop:
            return False

        if self.position == "buy":
            new_stop = row["bid"] - self.trailing_distance * 0.001
            if new_stop > self.trailing_stop_price:
                # Only update if new stop is better than current
                self.trailing_stop_price = new_stop
                if self.trailing_stop_price > self.stop_loss_price:
                    old_stop = self.stop_loss_price
                    self.stop_loss_price = self.trailing_stop_price
                    print(f"Trailing stop updated: {old_stop:.3f} → {self.stop_loss_price:.3f}")
                    return True
        else:  # sell
            new_stop = row["ask"] + self.trailing_distance * 0.001
            if new_stop < self.trailing_stop_price:
                # Only update if new stop is better than current
                self.trailing_stop_price = new_stop
                if self.trailing_stop_price < self.stop_loss_price:
                    old_stop = self.stop_loss_price
                    self.stop_loss_price = self.trailing_stop_price
                    print(f"Trailing stop updated: {old_stop:.3f} → {self.stop_loss_price:.3f}")
                    return True
        return False

    def check_breakeven_stop(self, row):
        """Move stop loss to breakeven when profit exceeds trigger threshold"""
        if not self.use_breakeven_stop or self.position is None:
            return False
            
        # Calculate current profit in points
        if self.position == "buy":
            current_profit_points = (row["bid"] - self.entry_price) * 1000
        else:  # sell
            current_profit_points = (self.entry_price - row["ask"]) * 1000
            
        # Calculate threshold in points (based on SL distance × breakeven_trigger_ratio)
        if self.use_dynamic_exits:
            sl_distance_points = abs(self.entry_price - self.stop_loss_price) * 1000
        else:
            sl_distance_points = self.stop_loss_points
            
        breakeven_threshold = sl_distance_points * self.breakeven_trigger_ratio
        
        # If profit exceeds threshold, move SL to entry
        if current_profit_points > breakeven_threshold:
            if self.position == "buy" and self.stop_loss_price < self.entry_price:
                self.stop_loss_price = self.entry_price
                print(f"Stop loss moved to breakeven: {self.entry_price:.3f}")
                self.signal_counts['breakeven_activated'] += 1
                return True
            elif self.position == "sell" and self.stop_loss_price > self.entry_price:
                self.stop_loss_price = self.entry_price
                print(f"Stop loss moved to breakeven: {self.entry_price:.3f}")
                self.signal_counts['breakeven_activated'] += 1
                return True
                
        return False

    def check_time_based_exit(self, current_time):
        """Check if trade should be closed based on maximum duration"""
        if self.position is None or self.entry_time is None:
            return False
            
        # Calculate trade duration in minutes
        duration = (current_time - self.entry_time).total_seconds() / 60
        
        # Exit if trade duration exceeds maximum
        if duration > self.max_trade_duration_minutes:
            print(f"Time-based exit: {duration:.0f} min exceeded maximum {self.max_trade_duration_minutes} min")
            self.signal_counts['time_exit'] += 1
            return True
            
        return False

    def check_partial_take_profit(self, row):
        """Check if partial take profit should be executed"""
        if not self.use_partial_take_profit or self.partial_exit_done or self.position is None:
            return False, 0
            
        if self.position == "buy":
            if row["bid"] >= self.partial_tp_price:
                # Calculate profit for partial position
                partial_lot = self.position_size * self.partial_tp_ratio
                points = (row["bid"] - self.entry_price) * 1000
                profit = points * partial_lot * 100
                
                # Reduce position size
                self.position_size -= partial_lot
                self.partial_exit_done = True
                
                print(f"Partial TP hit: {partial_lot:.2f} lots @ {row['bid']:.3f}, profit: {profit/100:.2f}$")
                self.signal_counts['partial_tp_hit'] += 1
                
                return True, profit
                
        elif self.position == "sell":
            if row["ask"] <= self.partial_tp_price:
                # Calculate profit for partial position
                partial_lot = self.position_size * self.partial_tp_ratio
                points = (self.entry_price - row["ask"]) * 1000
                profit = points * partial_lot * 100
                
                # Reduce position size
                self.position_size -= partial_lot
                self.partial_exit_done = True
                
                print(f"Partial TP hit: {partial_lot:.2f} lots @ {row['ask']:.3f}, profit: {profit/100:.2f}$")
                self.signal_counts['partial_tp_hit'] += 1
                
                return True, profit
                
        return False, 0

    def check_exit_conditions(self, row):
        """Check if trade should be closed based on various exit conditions"""
        if self.position is None:
            return False, 0, ""

        # Update trailing stop if enabled
        self.update_trailing_stop(row)
        
        # Check breakeven stop if enabled
        self.check_breakeven_stop(row)
        
        # Check time-based exit
        if self.check_time_based_exit(row.name):
            # Calculate current profit/loss
            if self.position == "buy":
                points = (row["bid"] - self.entry_price) * 1000
            else:  # sell
                points = (self.entry_price - row["ask"]) * 1000
                
            profit = points * self.position_size * 100
            return True, profit, "time exit"
        
        # Check partial take profit
        partial_hit, partial_profit = self.check_partial_take_profit(row)
        if partial_hit:
            # Update account but don't close position
            self.current_balance += partial_profit
            self.daily_profit_loss += partial_profit
            
        # Regular exit conditions
        c_bid = row["bid"]
        c_ask = row["ask"]

        if self.position == "buy":
            # stop-loss
            if c_bid <= self.stop_loss_price:
                points = (c_bid - self.entry_price) * 1000
                profit = points * self.position_size * 100
                return True, profit, "stop loss"
            # take-profit
            if c_bid >= self.take_profit_price:
                points = (c_bid - self.entry_price) * 1000
                profit = points * self.position_size * 100
                return True, profit, "take profit"

        else:  # sell
            if c_ask >= self.stop_loss_price:
                points = (self.entry_price - c_ask) * 1000
                profit = points * self.position_size * 100
                return True, profit, "stop loss"
            if c_ask <= self.take_profit_price:
                points = (self.entry_price - c_ask) * 1000
                profit = points * self.position_size * 100
                return True, profit, "take profit"

        return False, 0, ""

    def close_position(self, row, profit, reason):
        """Close the current position and update account/performance metrics"""
        exit_price = row["bid"] if self.position == "buy" else row["ask"]

        # Calculate trade duration
        duration_minutes = (row.name - self.entry_time).total_seconds() / 60 if self.entry_time else 0
        
        # Update account balance and performance tracking
        self.current_balance += profit
        self.daily_profit_loss += profit
        
        # Update consecutive loss counter
        if profit <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Create trade result record
        result = {
            "exit_time": row.name,
            "exit_price": exit_price,
            "profit": profit,
            "balance": self.current_balance,
            "exit_reason": reason,
            "duration_minutes": duration_minutes,
            "lot_size": self.original_position_size,
            "actual_r_multiple": profit / (self.original_position_size * self.stop_loss_points) if self.stop_loss_points > 0 else 0
        }

        # Add partial TP information if applicable
        if self.use_partial_take_profit and self.partial_exit_done:
            result["partial_tp_executed"] = True
            result["final_lot_size"] = self.position_size

        print(
            f"Close {self.position} @ {exit_price:.3f}  "
            f"p&l={profit/100:.2f}$  balance={self.current_balance/100:.2f}$  "
            f"duration={duration_minutes:.1f}min"
        )

        # reset position state
        self.position = None
        self.entry_price = self.stop_loss_price = self.take_profit_price = 0
        self.trailing_stop_price = 0
        self.entry_time = None
        self.partial_exit_done = False
        
        return result

    # ───────────────────────── back-test ───────────────────────────
    def backtest(self, csv_file, *, use_timeframe=None):
        """Run backtest on historical data with improved efficiency"""
        # Load and prepare data
        start_time = datetime.now()
        print(f"Starting backtest at {start_time.strftime('%H:%M:%S')}")
        
        data = self.load_data(csv_file)
        if data is None:
            return None

        if use_timeframe:
            data = self.resample_data(data, use_timeframe)

        df = self.prepare_data(data.copy())
        
        # Reset performance tracking
        self.trades.clear()
        self.equity_curve = [self.initial_capital]
        self.current_balance = self.initial_capital
        self.position = None
        self.consecutive_losses = 0
        self.daily_profit_loss = 0
        self.last_trade_date = None
        current_trade = None

        # Add ATR volatility information to track market conditions
        if 'atr' in df.columns:
            # Calculate rolling mean of ATR for comparison
            df['atr_mean_20'] = df['atr'].rolling(20).mean().fillna(method='bfill')
            
            # Mark volatile periods (ATR > 1.2 * rolling mean)
            df['high_volatility'] = df['atr'] > df['atr_mean_20'] * 1.2
            
            # Mark low volatility periods (ATR < 0.8 * rolling mean)
            df['low_volatility'] = df['atr'] < df['atr_mean_20'] * 0.8
        
        # Run the backtest using iterrows
        for idx, row in df.iterrows():
            # Update equity curve
            self.equity_curve.append(self.current_balance)

            # Check for exit if in a position
            if self.position is not None:
                exit_now, profit, reason = self.check_exit_conditions(row)
                if exit_now:
                    trade_res = self.close_position(row, profit, reason)
                    current_trade.update(trade_res)
                    self.trades.append(current_trade)
                    current_trade = None
            
            # Check for entry if not in a position
            if self.position is None:
                sig = self.check_entry_signals(row)
                if sig:
                    # Determine market regime directly from the row data
                    market_regime = "normal_volatility"
                    if 'high_volatility' in row and row['high_volatility']:
                        market_regime = "high_volatility"
                    elif 'low_volatility' in row and row['low_volatility']:
                        market_regime = "low_volatility"
                    
                    # Pass market regime to open position
                    current_trade = self.open_position(sig, row, market_regime)
                    self.trade_details[idx] = {
                        "type": sig,
                        "entry_price": self.entry_price,
                        "stop_loss": self.stop_loss_price,
                        "take_profit": self.take_profit_price,
                        "market_regime": market_regime
                    }

        # Force close at the end if still in a position
        if self.position is not None:
            last_row = df.iloc[-1]
            _, profit, _ = self.check_exit_conditions(last_row)
            trade_res = self.close_position(last_row, profit, "end of test")
            current_trade.update(trade_res)
            self.trades.append(current_trade)

        # Calculate final performance metrics
        results = self.calculate_performance()
        
        # Analyze performance by market regime
        if len(self.trades) > 0:
            regime_performance = {}
            for trade in self.trades:
                regime = trade.get("market_regime", "unknown")
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        "count": 0, 
                        "wins": 0, 
                        "losses": 0, 
                        "profit": 0
                    }
                
                regime_performance[regime]["count"] += 1
                if trade["profit"] > 0:
                    regime_performance[regime]["wins"] += 1
                else:
                    regime_performance[regime]["losses"] += 1
                regime_performance[regime]["profit"] += trade["profit"]
            
            # Calculate win rates and averages by regime
            for regime in regime_performance:
                count = regime_performance[regime]["count"]
                wins = regime_performance[regime]["wins"]
                
                if count > 0:
                    regime_performance[regime]["win_rate"] = (wins / count) * 100
                    regime_performance[regime]["avg_profit"] = regime_performance[regime]["profit"] / count
                else:
                    regime_performance[regime]["win_rate"] = 0
                    regime_performance[regime]["avg_profit"] = 0
                    
            results["regime_performance"] = regime_performance
        
        # Report execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Backtest completed in {duration:.2f} seconds")
        
        # Print regime performance summary
        if "regime_performance" in results:
            print("\n=== Performance by Market Regime ===")
            for regime, stats in results["regime_performance"].items():
                print(f"{regime}: {stats['count']} trades, {stats['win_rate']:.1f}% win rate, ${stats['profit']/100:.2f} profit")
        
        return results

    def backtest_on_prepared_data(self, df):
        """
        Run backtest directly on prepared DataFrame for optimization efficiency
        """
        # Reset performance tracking
        self.trades.clear()
        self.equity_curve = [self.initial_capital]
        self.current_balance = self.initial_capital
        self.position = None
        self.consecutive_losses = 0
        self.daily_profit_loss = 0
        self.last_trade_date = None
        current_trade = None

        # Run the backtest
        for idx, row in df.iterrows():
            # Update equity curve
            self.equity_curve.append(self.current_balance)

            # Check for exit if in a position
            if self.position is not None:
                exit_now, profit, reason = self.check_exit_conditions(row)
                if exit_now:
                    trade_res = self.close_position(row, profit, reason)
                    current_trade.update(trade_res)
                    self.trades.append(current_trade)
                    current_trade = None
            
            # Check for entry if not in a position
            if self.position is None:
                sig = self.check_entry_signals(row)
                if sig:
                    current_trade = self.open_position(sig, row)
                    self.trade_details[idx] = {
                        "type": sig,
                        "entry_price": self.entry_price,
                        "stop_loss": self.stop_loss_price,
                        "take_profit": self.take_profit_price,
                    }

        # Force close at the end if still in a position
        if self.position is not None and len(df) > 0:
            last_row = df.iloc[-1]
            _, profit, _ = self.check_exit_conditions(last_row)
            trade_res = self.close_position(last_row, profit, "end of test")
            current_trade.update(trade_res)
            self.trades.append(current_trade)

    # ────────────────────────── metrics ────────────────────────────
    def calculate_performance(self):
        """Calculate comprehensive performance metrics from trade history"""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_profit_cents": 0,
                "total_profit_usd": 0,
                "max_drawdown_percent": 0,
                "sharpe_ratio": 0,
                "avg_win_cents": 0,
                "avg_loss_cents": 0,
                "largest_win_cents": 0,
                "largest_loss_cents": 0,
                "avg_trade_duration": 0,
            }

        # Basic trade metrics
        tot = len(self.trades)
        wins = sum(t["profit"] > 0 for t in self.trades)
        losses = tot - wins
        win_rate = wins / tot * 100 if tot > 0 else 0

        # Profit metrics
        gross_profit = sum(t["profit"] for t in self.trades if t["profit"] > 0)
        gross_loss = abs(sum(t["profit"] for t in self.trades if t["profit"] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        total_profit = gross_profit - gross_loss
        total_profit_usd = total_profit / 100

        # Average trade metrics
        avg_win = gross_profit / wins if wins else 0
        avg_loss = gross_loss / losses if losses else 0
        largest_win = max((t["profit"] for t in self.trades if t["profit"] > 0), default=0)
        largest_loss = min((t["profit"] for t in self.trades if t["profit"] <= 0), default=0)

        # Trade duration analysis
        durations = [t.get("duration_minutes", 0) for t in self.trades]
        avg_dur = sum(durations) / len(durations) if durations else 0
        
        # Calculate max duration and frequency distribution
        max_duration = max(durations) if durations else 0
        duration_bins = {'<15min': 0, '15-60min': 0, '1-4hr': 0, '4-8hr': 0, '>8hr': 0}
        
        for dur in durations:
            if dur < 15:
                duration_bins['<15min'] += 1
            elif dur < 60:
                duration_bins['15-60min'] += 1
            elif dur < 240:
                duration_bins['1-4hr'] += 1
            elif dur < 480:
                duration_bins['4-8hr'] += 1
            else:
                duration_bins['>8hr'] += 1

        # Drawdown calculation
        peak = self.initial_capital
        drawdowns = []
        for bal in self.equity_curve:
            if bal > peak:
                peak = bal
            drawdowns.append((peak - bal) / peak * 100 if peak else 0)
        max_dd = max(drawdowns)
        
        # Calculate equity curve metrics
        if len(self.equity_curve) > 1:
            # Daily returns approximation
            rets = [
                self.equity_curve[i] / self.equity_curve[i - 1] - 1
                for i in range(1, len(self.equity_curve))
            ]
            
            # Sharpe ratio (annualized)
            sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252) if np.std(rets) > 0 else 0
            
            # Sortino ratio (only using negative returns for denominator)
            neg_rets = [r for r in rets if r < 0]
            sortino = (np.mean(rets) / np.std(neg_rets)) * np.sqrt(252) if neg_rets and np.std(neg_rets) > 0 else 0
            
            # Maximum consecutive wins/losses
            current_streak = 1
            max_win_streak = 0
            max_loss_streak = 0
            
            for i in range(1, len(self.trades)):
                if (self.trades[i]["profit"] > 0) == (self.trades[i-1]["profit"] > 0):
                    current_streak += 1
                else:
                    # If previous trade was a win, update max win streak
                    if self.trades[i-1]["profit"] > 0:
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, current_streak)
                    current_streak = 1
                    
            # Check final streak
            if self.trades and len(self.trades) > 0:
                if self.trades[-1]["profit"] > 0:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
        else:
            sharpe = 0
            sortino = 0
            max_win_streak = 0
            max_loss_streak = 0

        # Return comprehensive performance report
        return {
            "total_trades": tot,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_profit_cents": total_profit,
            "total_profit_usd": total_profit_usd,
            "total_return_percent": total_profit / self.initial_capital * 100,
            "max_drawdown_percent": max_dd,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "avg_win_cents": avg_win,
            "avg_loss_cents": avg_loss,
            "largest_win_cents": largest_win,
            "largest_loss_cents": largest_loss,
            "avg_trade_duration": avg_dur,
            "max_trade_duration": max_duration,
            "duration_distribution": duration_bins,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "daily_results": self.daily_results,
            "signal_counts": self.signal_counts,
            "risk_reward_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            "expectancy_per_trade": (win_rate/100 * avg_win - (1-win_rate/100) * avg_loss) if tot > 0 else 0,
            "average_r_multiple": np.mean([t.get("actual_r_multiple", 0) for t in self.trades]) if self.trades else 0
        }

    # ────────────────────────── plotting ───────────────────────────
    def plot_results(self):
        """Generate detailed performance charts"""
        if not self.trades or len(self.equity_curve) <= 1:
            print("Not enough data to plot.")
            return

        # Use helper functions from plotting module
        plot_equity_curve(self.equity_curve, self.trades)
        plot_monthly_performance(self.daily_results)
        plot_trade_analysis(self.trades, self.calculate_performance())

    # ───────────────────────── reporting ───────────────────────────
    def generate_trade_report(self, output_file=None):
        """Generate comprehensive trade report with detailed analysis"""
        if not self.trades:
            print("No trades to report.")
            return

        # Generate report using the utility function
        report = generate_performance_report(self)
        
        # Write to file or print
        # Write to file or print
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        else:
            print(report)

    def get_trade_date(self, timestamp):
        """
        Extract trade date for daily tracking
        """
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.date()
        return timestamp.date() if hasattr(timestamp, 'date') else None
        
    # ─────────────────────── optimization ──────────────────────────
    def optimize_parameters(
        self,
        csv_file,
        param_ranges,
        *,
        use_timeframe=None,
        num_combinations=10,
        optimization_metric="profit_factor",
        use_walk_forward=False,
        train_test_split=0.7
    ):
        """
        Optimize strategy parameters with improved efficiency and walk-forward testing
        
        Parameters:
        ----------
        csv_file : str
            Path to the data file
        param_ranges : dict
            Dictionary of parameter names and their possible values
        use_timeframe : str, optional
            Timeframe to resample data to
        num_combinations : int
            Maximum number of parameter combinations to test
        optimization_metric : str
            Metric to optimize ("profit_factor", "sharpe_ratio", etc.)
        use_walk_forward : bool
            Whether to use walk-forward validation
        train_test_split : float
            Ratio for train/test split if using walk_forward
            
        Returns:
        -------
        list
            Sorted list of parameter combinations and their performance
        """
        import itertools
        import random
        from tqdm import tqdm
        import concurrent.futures
        import time

        start_time = time.time()
        print(f"Optimization started at {time.strftime('%H:%M:%S')}")

        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
            
        print(f"Parameter space contains {total_combinations} combinations")
        
        # Generate combinations
        combos = list(itertools.product(*param_values))
        
        # Sample if too many combinations
        if len(combos) > num_combinations:
            print(f"Sampling {num_combinations} combinations from parameter space")
            combos = random.sample(combos, num_combinations)

        # Load data once for efficiency
        print("Loading data...")
        data = self.load_data(csv_file)
        if data is None:
            return []
            
        if use_timeframe:
            data = self.resample_data(data, use_timeframe)
            
        # Split data for walk-forward validation if requested
        if use_walk_forward:
            split_idx = int(len(data) * train_test_split)
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            print(f"Data split: {len(train_data)} rows training, {len(test_data)} rows testing")

        # Define test function for parallel processing
        def test_parameters(combo):
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            # Create bot instance with these parameters
            bot = XAUUSDCentScalpingBot(initial_capital=self.initial_capital, **params)
            
            if use_walk_forward:
                # Run on training data
                train_df = bot.prepare_data(train_data.copy())
                bot.backtest_on_prepared_data(train_df)
                train_perf = bot.calculate_performance()
                
                # Only test on test data if training results are promising
                if train_perf.get(optimization_metric, 0) > 1.0:  # Only continue if profitable on training
                    # Run on test data
                    test_df = bot.prepare_data(test_data.copy())
                    bot.backtest_on_prepared_data(test_df)
                    test_perf = bot.calculate_performance()
                    
                    # Calculate robustness score (how well test results match training)
                    # Higher is better - ideally close to 1.0 (test performs like training)
                    robustness = test_perf.get(optimization_metric, 0) / max(train_perf.get(optimization_metric, 0.001), 0.001)
                    
                    return {
                        "parameters": params,
                        "train_metrics": {
                            "profit_factor": train_perf.get("profit_factor", 0),
                            "win_rate": train_perf.get("win_rate", 0),
                            "total_profit_usd": train_perf.get("total_profit_usd", 0),
                            "max_drawdown_percent": train_perf.get("max_drawdown_percent", 0),
                            "sharpe_ratio": train_perf.get("sharpe_ratio", 0),
                        },
                        "test_metrics": {
                            "profit_factor": test_perf.get("profit_factor", 0),
                            "win_rate": test_perf.get("win_rate", 0),
                            "total_profit_usd": test_perf.get("total_profit_usd", 0),
                            "max_drawdown_percent": test_perf.get("max_drawdown_percent", 0),
                            "sharpe_ratio": test_perf.get("sharpe_ratio", 0),
                        },
                        "robustness": robustness,
                        "combined_score": (train_perf.get(optimization_metric, 0) + test_perf.get(optimization_metric, 0)) / 2 * robustness
                    }
                else:
                    # Skip test data if training results are poor
                    return {
                        "parameters": params,
                        "train_metrics": {
                            "profit_factor": train_perf.get("profit_factor", 0),
                            "win_rate": train_perf.get("win_rate", 0),
                            "total_profit_usd": train_perf.get("total_profit_usd", 0),
                            "max_drawdown_percent": train_perf.get("max_drawdown_percent", 0),
                            "sharpe_ratio": train_perf.get("sharpe_ratio", 0),
                        },
                        "test_metrics": {},
                        "robustness": 0,
                        "combined_score": 0
                    }
            else:
                # Standard optimization on full dataset
                prepared_df = bot.prepare_data(data.copy())
                bot.backtest_on_prepared_data(prepared_df)
                perf = bot.calculate_performance()
                
                return {
                    "parameters": params,
                    "metrics": {
                        "profit_factor": perf.get("profit_factor", 0),
                        "win_rate": perf.get("win_rate", 0),
                        "total_profit_usd": perf.get("total_profit_usd", 0),
                        "max_drawdown_percent": perf.get("max_drawdown_percent", 0),
                        "sharpe_ratio": perf.get("sharpe_ratio", 0),
                        "sortino_ratio": perf.get("sortino_ratio", 0),
                        "expectancy_per_trade": perf.get("expectancy_per_trade", 0),
                        "total_trades": perf.get("total_trades", 0)
                    },
                    "score": perf.get(optimization_metric, 0)
                }
                
        # Run optimization with parallel processing
        results = []
        print(f"Testing {len(combos)} parameter combinations using parallel processing...")
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_combo = {executor.submit(test_parameters, combo): combo for combo in combos}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_combo), total=len(combos)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    print(f"Parameter test generated an exception: {exc}")
        
        # Sort results by appropriate metric
        if use_walk_forward:
            results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        else:
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
        # Report execution time
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
            
        # Print top results
        print("\n=== Top Parameter Sets ===")
        n_to_show = min(5, len(results))
        
        if use_walk_forward:
            for i, r in enumerate(results[:n_to_show], 1):
                print(f"{i}. Train PF={r['train_metrics']['profit_factor']:.2f}, "
                      f"Test PF={r['test_metrics'].get('profit_factor', 0):.2f}, "
                      f"Robustness={r.get('robustness', 0):.2f}, "
                      f"Combined={r.get('combined_score', 0):.2f}")
                print(f"   Params: {r['parameters']}")
        else:
            for i, r in enumerate(results[:n_to_show], 1):
                metrics = r["metrics"]
                print(f"{i}. PF={metrics['profit_factor']:.2f}  Win={metrics['win_rate']:.2f}%  "
                      f"P&L=${metrics['total_profit_usd']:.2f}  DD={metrics['max_drawdown_percent']:.2f}%  "
                      f"Trades={metrics['total_trades']}")
                print(f"   Params: {r['parameters']}")
                      
        return results