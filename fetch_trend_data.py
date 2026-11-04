"""Download every candle file the trend test needs, in one go.

Usage (mean-reversion folder, venv active):
    python fetch_trend_data.py

Saves candles_coinbase_<coin>_usd_<1d or 4h>.csv for each coin below,
from START_DATE to now. Takes a few minutes: the 4h files are built
from 1h candles, about 150 requests per coin.
"""

import fetch_candles

START_DATE = "2021-10-01"
COINS = ["CRV", "BTC", "ETH", "UNI", "DOGE", "CAKE"]
TIMEFRAMES = ["1d", "4h"]


def jobs():
    """The fetch_candles.py arguments for every file, in order."""
    return [[START_DATE, "coinbase", f"{coin}/USD", "now", tf]
            for coin in COINS for tf in TIMEFRAMES]


def main():
    for args in jobs():
        fetch_candles.main(args)
        print()


if __name__ == "__main__":
    main()
