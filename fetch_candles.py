"""Download historical candles from an exchange and save them to a CSV.

Usage (from the mean-reversion folder, venv active):
    python fetch_candles.py                         uses the settings below
    python fetch_candles.py 2023-10-01              start from that date
    python fetch_candles.py 2023-10-01 kraken       another exchange
    python fetch_candles.py 2023-10-01 coinbase BTC/USD   another coin
    python fetch_candles.py 2021-10-01 coinbase CRV/USD 2023-10-01
                                       stop at an end date (not included);
                                       the dates go in the file name
    python fetch_candles.py 2021-10-01 coinbase BTC/USD now 1d
                                       another candle size ("now" = no end date)

If the exchange doesn't offer a candle size (Coinbase has no 4h), it
downloads 1h candles and combines them: every 4 hours starting at
00:00 UTC become one 4h candle.

The CSV has one row per candle:
    timestamp, date, open, high, low, close, volume
timestamp is milliseconds since 1970 (what ccxt uses); date is the same
moment in readable UTC so you can eyeball the file.
"""

import csv
import sys
import time
from datetime import datetime, timezone

import ccxt

# --- settings -------------------------------------------------------------
EXCHANGE = "coinbase"
SYMBOL = "CRV/USD"
TIMEFRAME = "1h"
START_DATE = "2023-10-01"  # about 3 years of history
BATCH_LIMIT = 300  # Coinbase's maximum candles per request
# ---------------------------------------------------------------------------


def date_to_ms(date_text):
    """'2023-10-01' -> milliseconds at midnight UTC on that day."""
    day = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(day.timestamp() * 1000)


def ms_to_date(ms):
    """Milliseconds -> '2023-10-01 13:00' in UTC."""
    moment = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M")


def next_since(batch, since, tf_ms, limit):
    """Where the next request should start.

    After a batch with candles: one candle past the newest one we got
    (using the newest timestamp, not the last row, in case an exchange
    sends candles newest-first).
    After an empty batch (before the coin was listed, or an exchange
    outage): skip ahead one full window so the loop keeps moving instead
    of stopping early or asking for the same empty window forever.
    """
    if batch:
        return max(c[0] for c in batch) + tf_ms
    return since + limit * tf_ms


def merge_candles(candles):
    """Sort by timestamp and keep one candle per timestamp.

    Batches can overlap at their edges, so the same candle may arrive
    twice. If it does, the later copy wins.
    """
    by_time = {}
    for candle in candles:
        by_time[candle[0]] = candle
    return [by_time[ts] for ts in sorted(by_time)]


def drop_unfinished(candles, tf_ms, now_ms):
    """Remove the candle that is still forming.

    A 1h candle that opened at 14:00 isn't final until 15:00. A backtest
    should only see finished candles, otherwise its last row changes every
    time you re-run it.
    """
    return [c for c in candles if c[0] + tf_ms <= now_ms]


def fetch_all(exchange, symbol, timeframe, since_ms, now_ms, limit=BATCH_LIMIT):
    """Fetch every candle from since_ms up to now_ms, one batch at a time."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    candles = []
    since = since_ms
    while since < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        # Ignore anything older than we asked for (some exchanges, like
        # Kraken, ignore 'since' and send their most recent candles).
        batch = [c for c in batch if c[0] >= since]
        candles.extend(batch)
        new_since = next_since(batch, since, tf_ms, limit)
        if batch:
            print(f"  got {len(batch):3} candles up to {ms_to_date(batch[-1][0])}")
        since = new_since
    candles = merge_candles(candles)
    return drop_unfinished(candles, tf_ms, now_ms)


def combine_candles(candles, tf_ms):
    """Merge smaller candles into bigger ones, e.g. 1h into 4h.

    Each candle goes in the block that starts at a multiple of tf_ms
    (for 4h: 00:00, 04:00, 08:00 ... UTC). For each block:
    open = first open, high = highest high, low = lowest low,
    close = last close, volume = total. A block with missing hours
    (an exchange outage) is still made from the hours that exist.
    Candles must be sorted oldest first.
    """
    combined = []
    for ts, o, h, l, c, v in candles:
        start = ts // tf_ms * tf_ms
        if combined and combined[-1][0] == start:
            block = combined[-1]
            block[2] = max(block[2], h)
            block[3] = min(block[3], l)
            block[4] = c
            block[5] += v
        else:
            combined.append([start, o, h, l, c, v])
    return combined


def fetch_timeframe(exchange, symbol, timeframe, since_ms, end_ms):
    """fetch_all, but builds the candle size from 1h if the exchange lacks it."""
    if timeframe in exchange.timeframes:
        return fetch_all(exchange, symbol, timeframe, since_ms, end_ms)
    print(f"  {timeframe} not offered here: fetching 1h and combining")
    hourly = fetch_all(exchange, symbol, "1h", since_ms, end_ms)
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    # the newest block may still be forming even if its hours are done
    return drop_unfinished(combine_candles(hourly, tf_ms), tf_ms, end_ms)


def write_csv(candles, path):
    """Save candles to a CSV with a header row."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "date", "open", "high", "low", "close", "volume"])
        for ts, o, h, l, c, v in candles:
            writer.writerow([ts, ms_to_date(ts), o, h, l, c, v])


def csv_name(exchange_id, symbol, timeframe, start_date=None, end_date=None):
    """('coinbase', 'CRV/USD', '1h') -> 'candles_coinbase_crv_usd_1h.csv'

    With an end date, both dates go in the name so it doesn't replace the
    up-to-today file:
    -> 'candles_coinbase_crv_usd_1h_2021-10-01_to_2023-10-01.csv'
    """
    pair = symbol.replace("/", "_").lower()
    if end_date is None:
        return f"candles_{exchange_id}_{pair}_{timeframe}.csv"
    return f"candles_{exchange_id}_{pair}_{timeframe}_{start_date}_to_{end_date}.csv"


def main(args):
    start_date = args[0] if len(args) > 0 else START_DATE
    exchange_id = args[1] if len(args) > 1 else EXCHANGE
    symbol = args[2] if len(args) > 2 else SYMBOL
    end_date = args[3] if len(args) > 3 and args[3] != "now" else None
    timeframe = args[4] if len(args) > 4 else TIMEFRAME

    exchange = getattr(ccxt, exchange_id)()  # public data, no API keys needed
    now_ms = int(time.time() * 1000)
    # stop at the end date (candles that open before it), but never past now
    end_ms = now_ms if end_date is None else min(date_to_ms(end_date), now_ms)
    until = "today" if end_date is None else end_date
    print(f"Fetching {symbol} {timeframe} from {exchange_id}, {start_date} to {until}")

    candles = fetch_timeframe(exchange, symbol, timeframe, date_to_ms(start_date), end_ms)
    if not candles:
        print("No candles came back. Check the symbol, exchange and date.")
        return

    path = csv_name(exchange_id, symbol, timeframe, start_date, end_date)
    write_csv(candles, path)
    print(f"Saved {len(candles)} candles to {path}")
    print(f"  first: {ms_to_date(candles[0][0])}   last: {ms_to_date(candles[-1][0])}")


if __name__ == "__main__":
    main(sys.argv[1:])
