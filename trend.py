"""Trend following test: textbook rules, no tuning, run once on all coins.

Usage (mean-reversion folder, venv active, after fetch_trend_data.py):
    python trend.py

Long only, one position per coin: IN (holding) while the rule says the
trend is up, OUT (cash) otherwise. Decisions at each candle's close,
fills at the next candle's open, $100 per trade, same fees as backtest.py.

Rules:
  donchian N/M  buy when the close is above the highest close of the
                previous N candles; sell when it's below the lowest
                close of the previous M candles.
  ma F/S        in while the F-candle average is above the S-candle
                average; out when it isn't.

The list and the pass rule were fixed on Nov 3, 2026, BEFORE running any
of it. Same candle counts on 1d and 4h (so 4h tests faster versions).
Don't change them after seeing results.
"""

import os
import statistics

from backtest import FEES, HOUR_MS, read_candles, trade_profit
from holdout import profit_factor

# --- fixed in advance (Nov 3, 2026) ----------------------------------------
COINS = ["CRV", "BTC", "ETH", "UNI", "DOGE", "CAKE"]
TIMEFRAMES = ["1d", "4h"]
RULES = [
    {"name": "donchian 20/10", "kind": "donchian", "entry": 20, "exit": 10},
    {"name": "donchian 55/20", "kind": "donchian", "entry": 55, "exit": 20},
    {"name": "ma 20/100", "kind": "ma", "fast": 20, "slow": 100},
    {"name": "ma 50/200", "kind": "ma", "fast": 50, "slow": 200},
]
PASS_FEES = "kraken taker"
PASS_MIN_TRADES = 30  # all coins together
PASS_PROFIT_FACTOR = 1.1  # all coins together, must be ABOVE this
# ---------------------------------------------------------------------------


def donchian_entry(closes, i, n):
    """Close above the highest of the previous n closes (not counting i)."""
    if i < n:
        return False
    return closes[i] > max(closes[i - n : i])


def donchian_exit(closes, i, m):
    """Close below the lowest of the previous m closes (not counting i)."""
    if i < m:
        return False
    return closes[i] < min(closes[i - m : i])


def ma_up(closes, i, fast, slow):
    """Fast average above slow average, both ending at i.
    None until there are `slow` closes (trend unknown)."""
    if i < slow - 1:
        return None
    fast_avg = statistics.fmean(closes[i - fast + 1 : i + 1])
    slow_avg = statistics.fmean(closes[i - slow + 1 : i + 1])
    return fast_avg > slow_avg


def make_rule(rule):
    """(should_buy, should_sell) functions of (closes, i) for one RULES entry."""
    if rule["kind"] == "donchian":
        return (lambda c, i: donchian_entry(c, i, rule["entry"]),
                lambda c, i: donchian_exit(c, i, rule["exit"]))
    if rule["kind"] == "ma":
        return (lambda c, i: ma_up(c, i, rule["fast"], rule["slow"]) is True,
                lambda c, i: ma_up(c, i, rule["fast"], rule["slow"]) is False)
    raise ValueError(f"unknown rule kind: {rule['kind']}")


def find_rule_trades(candles, should_buy, should_sell, delay=1):
    """Same walk as backtest.find_trades, with any buy/sell rule.

    delay = how many candles after the signal the order fills: 1 = the
    next open (normal), 2 = one candle late (a missed candle live).

    Returns (trades, open_trade). A position still held at the end isn't
    a trade yet; it's returned separately.
    """
    closes = [c["close"] for c in candles]
    trades = []
    position = None
    for i in range(len(candles) - delay):
        fill = candles[i + delay]
        if position is None:
            if should_buy(closes, i):
                position = {"entry_ts": fill["timestamp"], "entry_time": fill["date"],
                            "entry_price": fill["open"]}
        elif should_sell(closes, i):
            trades.append({
                "entry_ts": position["entry_ts"],
                "exit_ts": fill["timestamp"],
                "entry_time": position["entry_time"],
                "entry_price": position["entry_price"],
                "exit_time": fill["date"],
                "exit_price": fill["open"],
                "reason": "signal",
                "hours": (fill["timestamp"] - position["entry_ts"]) // HOUR_MS,
            })
            position = None
    return trades, position


def close_open_trade(trades, open_trade, candles):
    """Count a still-open position as sold at the last close.

    Trend trades can last months; leaving the current one out would hide
    the biggest trend of all if it's still running.
    """
    if open_trade is None:
        return trades
    last = candles[-1]
    return trades + [{
        "entry_ts": open_trade["entry_ts"],
        "exit_ts": last["timestamp"],
        "entry_time": open_trade["entry_time"],
        "entry_price": open_trade["entry_price"],
        "exit_time": last["date"],
        "exit_price": last["close"],
        "reason": "still open",
        "hours": (last["timestamp"] - open_trade["entry_ts"]) // HOUR_MS,
    }]


def profits(trades, buy_fee, sell_fee):
    return [trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee)
            for t in trades]


def hold_curve(candles, buy_fee, sell_fee, usd=100.0):
    """Running profit of buying $usd at the first open, valued at each close
    as if sold there (sell fee included). Used for hold's drawdown."""
    entry = candles[0]["open"]
    return [trade_profit(entry, c["close"], buy_fee, sell_fee, usd) for c in candles]


def strategy_curve(candles, trades, buy_fee, sell_fee, usd=100.0):
    """Running profit at each candle's close: closed trades so far, plus the
    open trade valued as if sold at this close. Same measuring stick as
    hold_curve, so the drawdowns compare fairly (a trade that gives back a
    big gain before exiting shows up here, not just its final result)."""
    curve = []
    closed = 0.0
    j = 0
    for c in candles:
        # trades that exited at this candle's open (or earlier) are done
        while j < len(trades) and trades[j]["exit_ts"] <= c["timestamp"]:
            t = trades[j]
            closed += trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee, usd)
            j += 1
        open_value = 0.0
        if j < len(trades) and trades[j]["entry_ts"] <= c["timestamp"]:
            open_value = trade_profit(trades[j]["entry_price"], c["close"],
                                      buy_fee, sell_fee, usd)
        curve.append(closed + open_value)
    return curve


def curve_drawdown(curve):
    """Biggest drop from a high point in a running-profit curve (starts at 0)."""
    peak = 0.0
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def passes(all_profits):
    """Fixed rule: at least 30 trades across all coins, profit factor above
    1.1, and a positive total."""
    return (len(all_profits) >= PASS_MIN_TRADES
            and profit_factor(all_profits) > PASS_PROFIT_FACTOR
            and sum(all_profits) > 0)


def candle_file(coin, timeframe):
    return f"candles_coinbase_{coin.lower()}_usd_{timeframe}.csv"


def run_timeframe(timeframe, folder="."):
    """Print both tables for one candle size. Returns False if files are missing."""
    paths = {coin: os.path.join(folder, candle_file(coin, timeframe)) for coin in COINS}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"{timeframe}: missing {', '.join(os.path.basename(p) for p in missing)}")
        print("  run python fetch_trend_data.py first")
        return False

    buy_fee, sell_fee = FEES[PASS_FEES]
    candles = {coin: read_candles(path) for coin, path in paths.items()}
    print(f"=== {timeframe} candles, {PASS_FEES} fees, $100 per trade ===")
    for coin in COINS:
        c = candles[coin]
        print(f"  {coin:5} {len(c):6} candles  {c[0]['date']} to {c[-1]['date']}")

    results = {}
    for rule in RULES:
        should_buy, should_sell = make_rule(rule)
        results[rule["name"]] = {}
        for coin in COINS:
            trades, open_trade = find_rule_trades(candles[coin], should_buy, should_sell)
            trades = close_open_trade(trades, open_trade, candles[coin])
            results[rule["name"]][coin] = {
                "profits": profits(trades, buy_fee, sell_fee),
                "drawdown": curve_drawdown(strategy_curve(candles[coin], trades,
                                                          buy_fee, sell_fee)),
            }

    print("\nTotal profit $ per coin, and all coins together:")
    print(f"{'rule':16}" + "".join(f"{coin:>9}" for coin in COINS)
          + f"{'trades':>8}{'total $':>9}{'PF':>6}  result")
    for rule in RULES:
        per_coin = results[rule["name"]]
        together = [p for coin in COINS for p in per_coin[coin]["profits"]]
        verdict = "PASS" if passes(together) else "fail"
        print(f"{rule['name']:16}"
              + "".join(f"{sum(per_coin[c]['profits']):>9.2f}" for c in COINS)
              + f"{len(together):>8}{sum(together):>9.2f}{profit_factor(together):>6.2f}"
              + f"  {verdict}")
    holds = {coin: hold_curve(candles[coin], buy_fee, sell_fee) for coin in COINS}
    print(f"{'buy and hold':16}" + "".join(f"{holds[c][-1]:>9.2f}" for c in COINS))

    print("\nMax drawdown $ per coin, measured at every close (smaller is better):")
    print(f"{'rule':16}" + "".join(f"{coin:>9}" for coin in COINS))
    for rule in RULES:
        per_coin = results[rule["name"]]
        print(f"{rule['name']:16}"
              + "".join(f"{per_coin[c]['drawdown']:>9.2f}" for c in COINS))
    print(f"{'buy and hold':16}" + "".join(f"{curve_drawdown(holds[c]):>9.2f}" for c in COINS))
    print()
    return True


def main():
    print(f"Pass rule (all coins together, {PASS_FEES}): at least {PASS_MIN_TRADES} "
          f"trades, profit factor > {PASS_PROFIT_FACTOR}, total > 0")
    print("A position still open at the end counts as sold at the last close.\n")
    for timeframe in TIMEFRAMES:
        run_timeframe(timeframe)


if __name__ == "__main__":
    main()
