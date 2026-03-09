# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get Binance Testnet API Keys

1. Go to https://testnet.binance.vision/
2. Sign up/login
3. Go to API Management
4. Create API key
5. Copy your API key and secret

## Step 3: Create .env File

Create a file named `.env` in the project root:

```env
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here
BINANCE_TESTNET=True
TRADING_PAIR=ETHUSDT
TIMEFRAME=4h
INITIAL_CAPITAL=1000.0
DAILY_LOSS_LIMIT_PERCENT=3.0
STOP_LOSS_ATR_MULTIPLIER=2.0
TRADING_START_HOUR=8
TRADING_END_HOUR=17
MAX_CONSECUTIVE_API_ERRORS=5
MAX_PRICE_MOVE_PERCENT=10.0
```

## Step 4: Run the Bot

```bash
python main.py
```

## Step 5: Monitor

- Watch the console for status updates
- Check `logs/trading_bot_YYYYMMDD.log` for detailed logs
- The bot will check for signals every 5 minutes

## Important Notes

- **Start with testnet**: Always test with `BINANCE_TESTNET=True` first
- **Trading hours**: Bot only trades 8 AM - 5 PM UTC
- **4-hour timeframe**: Signals are based on 4-hour candles
- **Stop-loss**: Monitored automatically (2x ATR below entry)
- **Daily limit**: Bot stops new trades if daily loss exceeds 3%

## Stopping the Bot

Press `Ctrl+C` to stop gracefully.
