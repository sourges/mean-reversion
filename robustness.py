"""Robustness checks for the two trend candidates: is the result sturdy?

Usage (mean-reversion folder, venv active, candle files from fetch_trend_data.py):
    python robustness.py

Each check breaks one thing on purpose and asks whether the profit survives:
  1 years      total still positive after removing the best year
  2 best trade total still positive after removing each coin's biggest winner
  3 neighbors  nearby speeds: all 3 positive, at least 2 with profit factor > 1.1
  4 high fees  still positive at Coinbase Adv 1 taker (0.75% each side)
  5 late       still positive when every order fills one candle late
A candidate SURVIVES only if it passes all five.

Checks, neighbors and pass rules were fixed on Nov 4, 2026, BEFORE running.
Don't change them after seeing results.
"""

import os

from backtest import FEES, read_candles, trade_profit
from holdout import profit_factor
from trend import COINS, candle_file, close_open_trade, find_rule_trades, make_rule

# --- fixed in advance (Nov 4, 2026) ----------------------------------------
CANDIDATES = [
    {
        "name": "1d donchian 20/10",
        "timeframe": "1d",
        "rule": {"kind": "donchian", "entry": 20, "exit": 10},
        "neighbors": [
            {"kind": "donchian", "entry": 15, "exit": 7},
            {"kind": "donchian", "entry": 25, "exit": 12},
            {"kind": "donchian", "entry": 30, "exit": 15},
        ],
    },
    {
        "name": "4h ma 50/200",
        "timeframe": "4h",
        "rule": {"kind": "ma", "fast": 50, "slow": 200},
        "neighbors": [
            {"kind": "ma", "fast": 40, "slow": 160},
            {"kind": "ma", "fast": 60, "slow": 240},
            {"kind": "ma", "fast": 50, "slow": 150},
        ],
    },
]
BASE_FEES = "kraken taker"
HIGH_FEES = "coinbase adv1 taker"
LATE_DELAY = 2  # fill 2 candles after the signal instead of 1
NEIGHBOR_PROFIT_FACTOR = 1.1
NEIGHBOR_MIN_GOOD = 2  # of 3
# ---------------------------------------------------------------------------


def load_coins(timeframe, folder="."):
    """{coin: candles} for every coin, or None if any file is missing."""
    paths = {coin: os.path.join(folder, candle_file(coin, timeframe)) for coin in COINS}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return {coin: read_candles(p) for coin, p in paths.items()}


def coin_trades(candles_by_coin, rule, delay=1):
    """{coin: trades} for one rule, an open trade counted at the last close."""
    buy, sell = make_rule(rule)
    result = {}
    for coin, candles in candles_by_coin.items():
        trades, open_trade = find_rule_trades(candles, buy, sell, delay)
        result[coin] = close_open_trade(trades, open_trade, candles)
    return result


def coin_profits(trades_by_coin, fee_name):
    """{coin: [profit per trade]} at one fee level."""
    buy_fee, sell_fee = FEES[fee_name]
    return {coin: [trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee)
                   for t in trades]
            for coin, trades in trades_by_coin.items()}


def all_profits(profits_by_coin):
    return [p for profits in profits_by_coin.values() for p in profits]


def profit_by_year(trades_by_coin, fee_name):
    """{year: total profit} by the year each trade closed (all coins together)."""
    buy_fee, sell_fee = FEES[fee_name]
    years = {}
    for trades in trades_by_coin.values():
        for t in trades:
            year = int(t["exit_time"][:4])
            profit = trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee)
            years[year] = years.get(year, 0.0) + profit
    return dict(sorted(years.items()))


def without_best_year(years):
    """Total profit after removing the best year's total."""
    if not years:
        return 0.0
    return sum(years.values()) - max(years.values())


def without_best_trades(profits_by_coin):
    """Total profit after removing each coin's biggest WINNING trade.
    A coin with no winners keeps all its trades."""
    total = 0.0
    for profits in profits_by_coin.values():
        total += sum(profits)
        if profits and max(profits) > 0:
            total -= max(profits)
    return total


def neighbors_pass(neighbor_profits):
    """All neighbors positive, and enough of them above the profit factor bar."""
    all_positive = all(sum(p) > 0 for p in neighbor_profits)
    good = sum(1 for p in neighbor_profits if profit_factor(p) > NEIGHBOR_PROFIT_FACTOR)
    return all_positive and good >= NEIGHBOR_MIN_GOOD


def rule_label(rule):
    if rule["kind"] == "donchian":
        return f"donchian {rule['entry']}/{rule['exit']}"
    return f"ma {rule['fast']}/{rule['slow']}"


def run_checks(candidate, candles_by_coin):
    """[(check name, detail text, passed)] for one candidate."""
    trades = coin_trades(candles_by_coin, candidate["rule"])
    base = coin_profits(trades, BASE_FEES)
    checks = []

    years = profit_by_year(trades, BASE_FEES)
    year_text = ", ".join(f"{y}: {v:.2f}" for y, v in years.items())
    rest = without_best_year(years)
    checks.append(("1 years", f"{year_text} | without best year: {rest:.2f}", rest > 0))

    rest = without_best_trades(base)
    checks.append(("2 best trade", f"without each coin's best trade: {rest:.2f}", rest > 0))

    neighbor_profits = []
    parts = []
    for rule in candidate["neighbors"]:
        p = all_profits(coin_profits(coin_trades(candles_by_coin, rule), BASE_FEES))
        neighbor_profits.append(p)
        parts.append(f"{rule_label(rule)}: {sum(p):.2f} (PF {profit_factor(p):.2f})")
    checks.append(("3 neighbors", " | ".join(parts), neighbors_pass(neighbor_profits)))

    high = sum(all_profits(coin_profits(trades, HIGH_FEES)))
    checks.append(("4 high fees", f"{HIGH_FEES}: {high:.2f}", high > 0))

    late_trades = coin_trades(candles_by_coin, candidate["rule"], delay=LATE_DELAY)
    late = sum(all_profits(coin_profits(late_trades, BASE_FEES)))
    checks.append(("5 late", f"one candle late: {late:.2f}", late > 0))
    return checks


def main(folder="."):
    for candidate in CANDIDATES:
        candles_by_coin = load_coins(candidate["timeframe"], folder)
        print(f"=== {candidate['name']} ({BASE_FEES} fees unless noted) ===")
        if candles_by_coin is None:
            print(f"  missing {candidate['timeframe']} candle files: "
                  f"run python fetch_trend_data.py first\n")
            continue
        base = all_profits(coin_profits(coin_trades(candles_by_coin, candidate["rule"]),
                                        BASE_FEES))
        print(f"  baseline: {len(base)} trades, total {sum(base):.2f}, "
              f"PF {profit_factor(base):.2f}   (should match trend.py)")
        checks = run_checks(candidate, candles_by_coin)
        for name, detail, passed in checks:
            print(f"  {'pass' if passed else 'FAIL'}  {name:13} {detail}")
        survived = all(passed for _, _, passed in checks)
        print(f"  -> {'SURVIVES' if survived else 'does not survive'} "
              f"({sum(p for _, _, p in checks)} of {len(checks)} checks)\n")


if __name__ == "__main__":
    main()
