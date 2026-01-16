# Daily GARCH Update System

## Quick Start

Run this command to set up automatic daily updates at 10 PM:

```bash
./schedule_garch.sh
```

## What It Does

- **Fetches latest market data** for all tickers in `holdings.csv`
- **Fits DCC-GARCH model** with 3 years of historical data
- **Generates 21-day forecasts** for returns (mu) and covariance
- **Caches results** to `.cache/garch_forecast.pkl` for fast app loading
- **Runs automatically** every day at 10:00 PM

## Manual Update

To run an update manually:

```bash
python3 update_garch.py
```

## Files Created

- `update_garch.py` - Main update script
- `schedule_garch.sh` - Cron job installer
- `garch_updates.log` - Detailed update logs
- `garch_cron.log` - Cron execution logs
- `.cache/garch_forecast.pkl` - Cached model and forecasts

## Monitoring

Check the logs:

```bash
# View recent updates
tail -50 garch_updates.log

# View cron execution
tail -50 garch_cron.log
```

View scheduled job:

```bash
crontab -l
```

## Removing the Schedule

To stop automatic updates:

```bash
crontab -l | grep -v 'update_garch.py' | crontab -
```

## Integration with App

The Streamlit app automatically loads cached forecasts from `.cache/garch_forecast.pkl` if available, falling back to real-time computation if cache is stale or missing.
