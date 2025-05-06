# Pattern-Based Trading Enhancement for XAUUSD Bot

This extension adds Japanese candlestick pattern recognition capabilities to the existing XAUUSD trading bot. It implements a comprehensive set of pattern identification techniques based on technical analysis principles.

## Features

- **Candlestick Pattern Recognition**: Identifies 13+ patterns including Marubozu, Hammer, Engulfing, Harami, Morning/Evening Star, and more
- **Support/Resistance Identification**: Automatically finds key price levels for target setting
- **Dynamic Stop Loss Placement**: Uses pattern-specific stop loss rules for better risk management
- **Pattern-Based Position Sizing**: Adjusts position size based on pattern confidence and historical performance
- **Risk-Averse Entry Mode**: Option to wait for confirmation after pattern formation
- **Performance Tracking**: Monitors success rate of each pattern type

## Integration Methods

There are two ways to use these pattern-based trading enhancements:

### 1. Use the Extended `XAUUSDPatternBot` Class

This class inherits from the original bot but adds pattern recognition capabilities:

```python
from xauusd_pattern_bot import XAUUSDPatternBot

# Create pattern-enhanced bot
bot = XAUUSDPatternBot(
    initial_capital=100000,
    risk_percent=0.5,
    # Standard parameters
    fast_ma_period=5,
    slow_ma_period=100,
    # ...other standard parameters...
    
    # Pattern-specific parameters
    use_pattern_recognition=True,  # Enable pattern recognition
    pattern_position_sizing=True,  # Enable pattern-based position sizing
    risk_averse_entry=True,        # Wait for confirmation after pattern
    min_body_pct=1.0,              # Min candle body size as % of price
    max_body_pct=10.0              # Max candle body size as % of price
)

# Use exactly as you would the original bot
performance = bot.backtest('xau.csv', use_timeframe='5T')
```

### 2. Enhance the Original Bot with Pattern Integration

If you prefer to keep using your existing bot instances, you can enhance them with pattern recognition:

```python
from xauusd_bot import XAUUSDCentScalpingBot
from pattern_integration import integrate_patterns_with_bot

# Create standard bot
bot = XAUUSDCentScalpingBot(initial_capital=100000, risk_percent=0.5)

# Enhance with pattern recognition
enhanced_bot = integrate_patterns_with_bot(bot)

# Use as normal
performance = enhanced_bot.backtest('xau.csv', use_timeframe='5T')
```

## Technical Analysis Rules

The implementation follows these key principles:

1. **Core Assumptions**
   - Markets discount everything; focus is on price action and volume
   - Only trade on days where candle body length is between 1% and 10% of price

2. **Pattern Types Implemented**
   - **Single-Candle**: Marubozu, Spinning Top, Doji, Hammer, Hanging Man, Shooting Star
   - **Two-Candle**: Engulfing, Harami, Piercing, Dark Cloud Cover
   - **Three-Candle**: Morning Star, Evening Star

3. **Position Sizing**
   - Base position sizing determined by original risk parameters
   - Modified based on pattern confidence (stronger patterns get larger positions)
   - Further adjusted by pattern historical performance

4. **Stop Loss Placement**
   - Each pattern has specific stop loss rules (e.g., below bar low for Hammer)
   - Risk per trade is still limited to 1-2% of equity

5. **Profit Targets**
   - Set using support/resistance levels identified from price action
   - Falls back to specified reward:risk ratio if no clear S/R levels

## Configuration Options

### Pattern Recognition Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_pattern_recognition` | Enable/disable pattern recognition | `True` |
| `pattern_position_sizing` | Adjust position size based on pattern confidence | `True` |
| `risk_averse_entry` | Wait for confirmation after pattern formation | `True` |
| `min_body_pct` | Minimum candle body size as % of price | `1.0` |
| `max_body_pct` | Maximum candle body size as % of price | `10.0` |

## Enhanced Reports

The pattern-enhanced bot generates additional reporting on pattern performance:

```
Pattern Recognition Analysis
--------------------------------------------------
Pattern recognition enabled: True
Pattern position sizing: True
Risk-averse entry mode: True

Pattern Performance:
bullish_engulfing     : 15 trades, 60.0% win rate (9/15)
morning_star          : 7 trades, 71.4% win rate (5/7)
bearish_engulfing     : 12 trades, 58.3% win rate (7/12)
...
```

## How It Works

1. The bot identifies candlestick patterns in the price data
2. Each pattern is evaluated for strength and historical performance
3. Patterns are combined with the existing technical indicators (MA, Stochastic)
4. When both approaches agree, it generates the strongest signals
5. Position size, stops, and targets are adjusted based on the specific pattern
6. Performance of each pattern type is tracked for continuous improvement

## Implementation Files

- `utils/candlestick_patterns.py`: Core pattern recognition functions
- `pattern_integration.py`: Functions to integrate patterns with existing bot
- `xauusd_pattern_bot.py`: Extended bot class with pattern capabilities
- `README_PATTERN_TRADING.md`: This documentation file