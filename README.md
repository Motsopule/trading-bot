# Crypto Trading Bot

A Python-based cryptocurrency trading bot for Binance that trades ETH/USDT using a 4-hour timeframe with a Moving Average crossover strategy.

## Features

- **Strategy**: MA crossover with entry when 50MA is above 200MA and 20MA crosses above 50MA, exit when 20MA crosses below 50MA
- **Risk Management**: 
  - Stop-loss at 2x ATR (Average True Range)
  - Daily loss limit (3% of capital)
  - Trading hours restriction (London/NY overlap: 8 AM - 5 PM UTC)
- **Kill Switch**: Automatic shutdown on consecutive API errors or large price moves
- **Comprehensive Logging**: All trades, errors, and events logged to files
- **Paper Trading**: Start with Binance testnet before live trading

## Project Structure

```
Trading Bot/
├── data.py          # Market data fetching from Binance API
├── strategy.py      # Trading strategy logic (MA crossover, ATR)
├── risk.py          # Risk management (loss limits, trading hours, kill switch)
├── execution.py     # Order placement and management
├── main.py          # Main bot orchestration
├── requirements.txt # Python dependencies
├── .env             # Environment variables (create from .env.example)
└── README.md        # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=True

# Trading Configuration
TRADING_PAIR=ETHUSDT
TIMEFRAME=4h
INITIAL_CAPITAL=1000.0

# Risk Management
DAILY_LOSS_LIMIT_PERCENT=3.0
STOP_LOSS_ATR_MULTIPLIER=2.0

# Trading Hours (UTC)
TRADING_START_HOUR=8
TRADING_END_HOUR=17

# Kill Switch Configuration
MAX_CONSECUTIVE_API_ERRORS=5
MAX_PRICE_MOVE_PERCENT=10.0
```

### 3. Get Binance API Credentials

#### For Testnet (Paper Trading):
1. Visit https://testnet.binance.vision/
2. Create an account and generate API keys
3. Set `BINANCE_TESTNET=True` in your `.env` file

#### For Live Trading:
1. Visit https://www.binance.com/
2. Go to API Management and create API keys
3. Set `BINANCE_TESTNET=False` in your `.env` file
4. **Important**: Start with small amounts and test thoroughly!

## Usage

### Running the Bot

```bash
python main.py
```

The bot will:
- Connect to Binance (testnet or live based on configuration)
- Fetch market data every 5 minutes
- Check for entry/exit signals based on the strategy
- Manage risk according to configured limits
- Log all activities to `logs/trading_bot_YYYYMMDD.log`

### Stopping the Bot

Press `Ctrl+C` to gracefully stop the bot. If a position is open, the bot will log a warning.

## Strategy Details

### Entry Conditions
1. 50-period MA is above the 200-period Moving Average
2. 20-period MA crosses above 50-period MA

### Exit Conditions
1. 20-period MA crosses below 50-period MA
2. Stop-loss triggered (2x ATR below entry price)

### Risk Management
- **Position Sizing**: Based on 1% risk per trade
- **Stop-Loss**: 2x ATR below entry price
- **Daily Loss Limit**: 3% of total capital (no new trades if hit)
- **Trading Hours**: Only trades during 8 AM - 5 PM UTC (London/NY overlap)

## Kill Switch

The bot will automatically stop trading if:
- **Consecutive API Errors**: 5 or more consecutive API errors
- **Large Price Moves**: Price moves more than 10% in a short period
- **Daily Loss Limit**: 3% daily loss limit is reached

When kill switch is activated:
- No new trades will be opened
- Optionally, open positions can be closed (configured in code)

## Logging

All activities are logged to:
- **File**: `logs/trading_bot_YYYYMMDD.log` (detailed logs)
- **Console**: Important events and status updates

Logs include:
- Trade entries and exits
- Order placements
- Risk management events
- API errors
- Kill switch activations

## Testing

### Paper Trading (Recommended)
1. Set `BINANCE_TESTNET=True` in `.env`
2. Run the bot and monitor for at least 100 simulated trades
3. Review logs and performance
4. Adjust strategy parameters if needed

### Live Trading
1. Only proceed after successful paper trading
2. Set `BINANCE_TESTNET=False` in `.env`
3. Start with small capital amounts
4. Monitor closely, especially in the first few days

## Risk Warning

**⚠️ IMPORTANT**: Cryptocurrency trading involves substantial risk. This bot is provided as-is for educational purposes. Always:
- Start with paper trading
- Use only capital you can afford to lose
- Monitor the bot regularly
- Understand the strategy before using it
- Test thoroughly before live trading

## Troubleshooting

### API Errors
- Check API keys are correct
- Verify API permissions (spot trading enabled)
- Check internet connection
- Review Binance API status

### Insufficient Data
- Ensure you have enough historical data (200+ candles)
- Check if Binance API is accessible
- Verify trading pair symbol is correct

### Kill Switch Activated
- Review logs for the reason
- Check API connectivity
- Verify market conditions
- Manually reset if needed (modify `risk_state.json`)

## Configuration Options

All configuration is done via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Binance API key | Required |
| `BINANCE_API_SECRET` | Binance API secret | Required |
| `BINANCE_TESTNET` | Use testnet | True |
| `TRADING_PAIR` | Trading pair | ETHUSDT |
| `TIMEFRAME` | Kline interval | 4h |
| `INITIAL_CAPITAL` | Starting capital | 1000.0 |
| `DAILY_LOSS_LIMIT_PERCENT` | Daily loss limit % | 3.0 |
| `STOP_LOSS_ATR_MULTIPLIER` | ATR multiplier for stop-loss | 2.0 |
| `TRADING_START_HOUR` | Trading start hour (UTC) | 8 |
| `TRADING_END_HOUR` | Trading end hour (UTC) | 17 |
| `MAX_CONSECUTIVE_API_ERRORS` | Max errors before kill switch | 5 |
| `MAX_PRICE_MOVE_PERCENT` | Max price move % before kill switch | 10.0 |

## License

This project is provided as-is for educational purposes. Use at your own risk.
