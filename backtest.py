"""Backtest the mean reversion strategy on a candle CSV from fetch_candles.py.

Usage (from the mean-reversion folder, venv active):
    python backtest.py                                   default candle file
    python backtest.py candles_coinbase_btc_usd_1h.csv   another file

How it trades (one position at a time, long only):
  - At each candle's close, compute z = (close - average) / std over the
    last WINDOW closes (only candles up to now, never future ones).
  - Not holding and z < ENTRY_Z         -> buy,
    but only if close > its TREND_WINDOW average (trend filter, if on).
  - Holding and z >= EXIT_Z             -> sell ("signal").
  - Holding and close <= entry * (1 - STOP_PCT)   -> sell ("stop").
  - Holding for MAX_HOURS or more       -> sell ("time").
  - Every buy and sell fills at the NEXT candle's open, because you can't
    trade at a close you've only just seen.

Profit is in USD for a fixed TRADE_USD per trade (no compounding).

The trades are the same at every fee level (fees don't change when the
strategy buys or sells), only the profit changes. So it saves one trades
CSV per fee level, named after the candle file and the fees:
    trades_coinbase_crv_usd_1h_kraken_taker.csv
          (candles from)        (fees used)
Each has a 'profit' column, so trade-analyzer's report.py reads it.
"""

import csv
import os
import statistics
import sys

# --- settings -------------------------------------------------------------
CANDLE_FILE = "candles_coinbase_crv_usd_1h.csv"
WINDOW = 20  # candles in the moving average
ENTRY_Z = -2.0  # buy when z drops below this
EXIT_Z = 0.0  # sell when z climbs back to this
STOP_PCT = None  # e.g. 0.08 = sell if close is 8% below entry; None = off
MAX_HOURS = None  # e.g. 48 = sell after 48 hours held; None = off
TREND_WINDOW = 200  # only buy dips when close is above this average; None = off
TRADE_USD = 100.0

# (buy fee, sell fee) as fractions: 0.0038 = 0.38%
FEES = {
    "kraken taker": (0.0038, 0.0038),
    "kraken maker": (0.0022, 0.0022),
    "coinbase adv1 taker": (0.0075, 0.0075),
    "coinbase adv1 maker": (0.0035, 0.0035),
    "coinbase adv2 taker": (0.0030, 0.0030),
    "coinbase adv2 maker": (0.0015, 0.0015),
}
# ---------------------------------------------------------------------------

HOUR_MS = 3_600_000


def read_candles(path):
    """Read a fetch_candles.py CSV into a list of dicts with numbers."""
    candles = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            candles.append({
                "timestamp": int(row["timestamp"]),
                "date": row["date"],
                "open": float(row["open"]),
                "close": float(row["close"]),
            })
    return candles


def zscore(closes, i, window):
    """z of closes[i] against the `window` closes ending at i (inclusive).

    Returns None when there aren't enough candles yet, or when every close
    in the window is the same (std of 0 means z can't be measured).
    """
    if i < window - 1:
        return None
    recent = closes[i - window + 1 : i + 1]
    std = statistics.pstdev(recent)
    if std == 0:
        return None
    return (closes[i] - statistics.fmean(recent)) / std


def above_trend(closes, i, trend_window):
    """True if closes[i] is above the average of the `trend_window` closes
    ending at i. False when there aren't enough candles yet (no trend known,
    so no buying)."""
    if i < trend_window - 1:
        return False
    recent = closes[i - trend_window + 1 : i + 1]
    return closes[i] > statistics.fmean(recent)


def exit_reason(z, close, entry_price, held_ms, exit_z, stop_pct, max_hours):
    """Why to sell at this candle's close, or None to keep holding.

    If several are true at once, 'signal' wins, then 'stop', then 'time'.
    """
    if z is not None and z >= exit_z:
        return "signal"
    if stop_pct is not None and close <= entry_price * (1 - stop_pct):
        return "stop"
    if max_hours is not None and held_ms >= max_hours * HOUR_MS:
        return "time"
    return None


def find_trades(candles, window, entry_z, exit_z=0.0, stop_pct=None, max_hours=None,
                trend_window=None):
    """Walk through the candles and return (trades, open_trade).

    trend_window (None = off) only filters buys: a dip counts only when the
    close is above its trend_window average. Exits ignore it.

    Each trade is a dict: entry_time, entry_price, exit_time, exit_price,
    reason, hours. open_trade is the position still held at the end
    (no exit yet), or None.
    """
    closes = [c["close"] for c in candles]
    trades = []
    position = None

    # The last candle can't trigger anything: there's no next open to fill at.
    for i in range(len(candles) - 1):
        z = zscore(closes, i, window)
        fill = candles[i + 1]  # where a decision made now gets filled

        if position is None:
            trend_ok = trend_window is None or above_trend(closes, i, trend_window)
            if z is not None and z < entry_z and trend_ok:
                position = {
                    "entry_ts": fill["timestamp"],
                    "entry_time": fill["date"],
                    "entry_price": fill["open"],
                }
        else:
            # held time is measured at this candle's close
            held_ms = candles[i]["timestamp"] + HOUR_MS - position["entry_ts"]
            reason = exit_reason(z, closes[i], position["entry_price"], held_ms,
                                 exit_z, stop_pct, max_hours)
            if reason:
                trades.append({
                    "entry_time": position["entry_time"],
                    "entry_price": position["entry_price"],
                    "exit_time": fill["date"],
                    "exit_price": fill["open"],
                    "reason": reason,
                    "hours": (fill["timestamp"] - position["entry_ts"]) // HOUR_MS,
                })
                position = None

    return trades, position


def trade_profit(entry_price, exit_price, buy_fee, sell_fee, usd=TRADE_USD):
    """USD profit for spending `usd` on a buy and selling everything.

    The buy fee comes out of what you get: coins = usd * (1 - buy_fee) / entry.
    The sell fee comes out of the proceeds: coins * exit * (1 - sell_fee).
    """
    coins = usd * (1 - buy_fee) / entry_price
    proceeds = coins * exit_price * (1 - sell_fee)
    return proceeds - usd


def summarize(trades, buy_fee, sell_fee, usd=TRADE_USD):
    """Count, win rate (%), total and average profit at one fee level."""
    profits = [trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee, usd)
               for t in trades]
    if not profits:
        return {"trades": 0, "win_rate": 0.0, "total": 0.0, "average": 0.0}
    wins = sum(1 for p in profits if p > 0)
    return {
        "trades": len(profits),
        "win_rate": wins / len(profits) * 100,
        "total": sum(profits),
        "average": sum(profits) / len(profits),
    }


def buy_and_hold(candles, buy_fee, sell_fee, usd=TRADE_USD):
    """Profit from buying at the first open and selling at the last close."""
    return trade_profit(candles[0]["open"], candles[-1]["close"], buy_fee, sell_fee, usd)


def write_trades(trades, path, buy_fee, sell_fee, usd=TRADE_USD):
    """Save one row per trade, with a 'profit' column for report.py."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entry_time", "entry_price", "exit_time", "exit_price",
                         "reason", "hours", "profit"])
        for t in trades:
            profit = trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee, usd)
            writer.writerow([t["entry_time"], t["entry_price"], t["exit_time"],
                             t["exit_price"], t["reason"], t["hours"], f"{profit:.4f}"])


def trades_name(candle_path, fee_name):
    """('candles_coinbase_crv_usd_1h.csv', 'kraken taker')
    -> 'trades_coinbase_crv_usd_1h_kraken_taker.csv'

    Any folder in candle_path is dropped (the file is saved where you run it).
    """
    stem = os.path.basename(candle_path)
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    if stem.startswith("candles_"):
        stem = stem[len("candles_") :]
    return f"trades_{stem}_{fee_name.replace(' ', '_')}.csv"


def print_table(trades, candles):
    """One row per fee level, with buy and hold for comparison."""
    print(f"{'fees':22}{'trades':>7}{'win %':>8}{'total $':>10}{'avg $':>8}{'hold $':>9}")
    for name, (buy_fee, sell_fee) in FEES.items():
        s = summarize(trades, buy_fee, sell_fee)
        hold = buy_and_hold(candles, buy_fee, sell_fee)
        print(f"{name:22}{s['trades']:>7}{s['win_rate']:>8.1f}{s['total']:>10.2f}"
              f"{s['average']:>8.3f}{hold:>9.2f}")


def count_exits(trades):
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return reasons


def on_off(value, unit):
    return "off" if value is None else f"{value}{unit}"


def main(args):
    path = args[0] if args else CANDLE_FILE
    candles = read_candles(path)
    print(f"{path}: {len(candles)} candles, {candles[0]['date']} to {candles[-1]['date']} UTC")
    stop_text = "off" if STOP_PCT is None else f"{STOP_PCT * 100:g}%"
    print(f"window {WINDOW}, entry z < {ENTRY_Z}, exit z >= {EXIT_Z}, "
          f"price stop {stop_text}, time stop {on_off(MAX_HOURS, 'h')}, "
          f"trend filter {on_off(TREND_WINDOW, ' candles')}, ${TRADE_USD:g} per trade")

    trades, open_trade = find_trades(candles, WINDOW, ENTRY_Z, EXIT_Z, STOP_PCT,
                                     MAX_HOURS, TREND_WINDOW)
    print(f"exits: {count_exits(trades)}")
    print()
    print_table(trades, candles)
    print("(hold $ = buying $100 at the first candle and selling at the last)")

    if TREND_WINDOW is not None:
        # same settings without the filter, so you can see what it changed
        unfiltered, _ = find_trades(candles, WINDOW, ENTRY_Z, EXIT_Z, STOP_PCT, MAX_HOURS)
        print(f"\nSame settings WITHOUT the trend filter (for comparison):")
        print_table(unfiltered, candles)

    if open_trade:
        print(f"\nStill holding from {open_trade['entry_time']} at "
              f"{open_trade['entry_price']} (not counted above)")

    print(f"\nSaved {len(trades)} trades, one file per fee level:")
    for name, (buy_fee, sell_fee) in FEES.items():
        trades_file = trades_name(path, name)
        write_trades(trades, trades_file, buy_fee, sell_fee)
        print(f"  {trades_file}")
    print("Analyze one with: python ..\\trade-analyzer\\report.py <trades file>")


if __name__ == "__main__":
    main(sys.argv[1:])
