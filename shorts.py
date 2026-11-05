"""Short side of 1d Donchian 20/10: does shorting breakdowns add an edge?

Usage (mean-reversion folder, venv active, 1d candle files from fetch_trend_data.py):
    python shorts.py

Mirror of the long rule, tested on spot prices (a 1x perp short makes about
what the spot price drop is, minus fees and funding):
  short  when a daily close is BELOW the lowest of the previous 20 closes
  cover  when a daily close is ABOVE the highest of the previous 10 closes
Fills at the next open, $100 per trade, 1x (no leverage).

Costs are deliberately PESSIMISTIC:
  fees     Kraken spot taker 0.38% each side (perp fees are usually lower)
  funding  shorts PAY 0.03% of the position per day (about 11% a year);
           in practice shorts usually RECEIVE funding
  liquidation  at 1x a short is wiped out if price doubles: if any daily
           close during the trade reaches 2x the entry, the trade is -$100

Pass rule and checks (same as the long side), all coins together:
  main test  at least 30 trades, profit factor > 1.1, total > 0
  1 years      total still positive after removing the best year
  2 best trade total still positive after removing each coin's biggest winner
  3 neighbors  15/7, 25/12, 30/15: all positive, at least 2 with PF > 1.1
  4 high fees  still positive at Coinbase Adv 1 taker (0.75% each side)
  5 late       still positive when every order fills one candle late
SURVIVES only if the main test and all five checks pass.

Fixed on Nov 5, 2026, BEFORE running. Don't change after seeing results.
"""

from backtest import FEES
from holdout import profit_factor
from robustness import (coin_profits, coin_trades, load_coins, neighbors_pass,
                        without_best_trades, without_best_year)
from trend import COINS, close_open_trade, find_rule_trades

# --- fixed in advance (Nov 5, 2026) ----------------------------------------
TIMEFRAME = "1d"
RULE = {"entry": 20, "exit": 10}
NEIGHBORS = [{"entry": 15, "exit": 7}, {"entry": 25, "exit": 12}, {"entry": 30, "exit": 15}]
BASE_FEES = "kraken taker"
HIGH_FEES = "coinbase adv1 taker"
FUNDING_PER_DAY = 0.0003  # shorts pay 0.03% of $100 per day held
LATE_DELAY = 2
PASS_MIN_TRADES = 30
PASS_PROFIT_FACTOR = 1.1
USD = 100.0
# ---------------------------------------------------------------------------

HOUR_MS = 3_600_000


def short_entry(closes, i, n):
    """Close below the lowest of the previous n closes (not counting i)."""
    if i < n:
        return False
    return closes[i] < min(closes[i - n : i])


def short_exit(closes, i, m):
    """Close above the highest of the previous m closes (not counting i)."""
    if i < m:
        return False
    return closes[i] > max(closes[i - m : i])


def short_rule(rule):
    """(should_open, should_cover) functions of (closes, i)."""
    return (lambda c, i: short_entry(c, i, rule["entry"]),
            lambda c, i: short_exit(c, i, rule["exit"]))


def short_profit(entry, exit_price, open_fee, cover_fee, hours, funding_per_day,
                 usd=USD, liquidated=False):
    """USD profit of a $usd 1x short.

    gain     usd * (entry - exit) / entry   (price falling is the profit)
    fees     open_fee on $usd, cover_fee on what the position is worth at exit
    funding  usd * funding_per_day for each day held
    A liquidated short loses the whole $usd.
    """
    if liquidated:
        return -usd
    gain = usd * (entry - exit_price) / entry
    fees = usd * open_fee + usd * exit_price / entry * cover_fee
    funding = usd * funding_per_day * hours / 24
    return gain - fees - funding


def was_liquidated(candles, trade):
    """True if any close while the short was open (from the entry candle up to
    the one before the cover fill), or the cover price, reached 2x the entry."""
    limit = 2 * trade["entry_price"]
    if trade["exit_price"] >= limit:
        return True
    return any(c["close"] >= limit for c in candles
               if trade["entry_ts"] <= c["timestamp"] < trade["exit_ts"])


def coin_short_trades(candles_by_coin, rule, delay=1):
    """{coin: trades}, a short still open counted as covered at the last close."""
    should_open, should_cover = short_rule(rule)
    result = {}
    for coin, candles in candles_by_coin.items():
        trades, open_trade = find_rule_trades(candles, should_open, should_cover, delay)
        result[coin] = close_open_trade(trades, open_trade, candles)
    return result


def coin_short_profits(trades_by_coin, candles_by_coin, fee_name, funding_per_day):
    """{coin: [profit per short]}."""
    open_fee, cover_fee = FEES[fee_name]
    result = {}
    for coin, trades in trades_by_coin.items():
        result[coin] = [short_profit(t["entry_price"], t["exit_price"], open_fee, cover_fee,
                                     t["hours"], funding_per_day,
                                     liquidated=was_liquidated(candles_by_coin[coin], t))
                        for t in trades]
    return result


def all_profits(profits_by_coin):
    return [p for profits in profits_by_coin.values() for p in profits]


def by_year(trades_by_coin, profits_by_coin):
    """{year: total} by the year each trade closed."""
    years = {}
    for coin, trades in trades_by_coin.items():
        for t, p in zip(trades, profits_by_coin[coin]):
            year = int(t["exit_time"][:4])
            years[year] = years.get(year, 0.0) + p
    return dict(sorted(years.items()))


def passes(profits):
    return (len(profits) >= PASS_MIN_TRADES
            and profit_factor(profits) > PASS_PROFIT_FACTOR
            and sum(profits) > 0)


def long_by_year(candles_by_coin):
    """The long side (1d Donchian 20/10, Kraken taker) by year, for comparison."""
    rule = {"kind": "donchian", "entry": RULE["entry"], "exit": RULE["exit"]}
    trades = coin_trades(candles_by_coin, rule)
    return by_year(trades, coin_profits(trades, BASE_FEES))


def year_rows(longs, shorts):
    """[(year, long, short, both)] for every year either side traded, then
    ('total', long, short, both)."""
    rows = []
    for year in sorted(set(longs) | set(shorts)):
        l, s = longs.get(year, 0.0), shorts.get(year, 0.0)
        rows.append((year, l, s, l + s))
    l, s = sum(longs.values()), sum(shorts.values())
    rows.append(("total", l, s, l + s))
    return rows


def run_checks(candles_by_coin):
    """[(check name, detail text, passed)], main test first."""
    trades = coin_short_trades(candles_by_coin, RULE)
    base = coin_short_profits(trades, candles_by_coin, BASE_FEES, FUNDING_PER_DAY)
    everything = all_profits(base)
    checks = [("main test",
               f"{len(everything)} trades, total {sum(everything):.2f}, "
               f"PF {profit_factor(everything):.2f}", passes(everything))]

    years = by_year(trades, base)
    rest = without_best_year(years)
    year_text = ", ".join(f"{y}: {v:.2f}" for y, v in years.items())
    checks.append(("1 years", f"{year_text} | without best year: {rest:.2f}", rest > 0))

    rest = without_best_trades(base)
    checks.append(("2 best trade", f"without each coin's best trade: {rest:.2f}", rest > 0))

    neighbor_profits = []
    parts = []
    for rule in NEIGHBORS:
        p = all_profits(coin_short_profits(coin_short_trades(candles_by_coin, rule),
                                           candles_by_coin, BASE_FEES, FUNDING_PER_DAY))
        neighbor_profits.append(p)
        parts.append(f"{rule['entry']}/{rule['exit']}: {sum(p):.2f} (PF {profit_factor(p):.2f})")
    checks.append(("3 neighbors", " | ".join(parts), neighbors_pass(neighbor_profits)))

    high = sum(all_profits(coin_short_profits(trades, candles_by_coin, HIGH_FEES,
                                              FUNDING_PER_DAY)))
    checks.append(("4 high fees", f"{HIGH_FEES}: {high:.2f}", high > 0))

    late_trades = coin_short_trades(candles_by_coin, RULE, delay=LATE_DELAY)
    late = sum(all_profits(coin_short_profits(late_trades, candles_by_coin, BASE_FEES,
                                              FUNDING_PER_DAY)))
    checks.append(("5 late", f"one candle late: {late:.2f}", late > 0))
    return checks, trades, base


def main(folder="."):
    candles_by_coin = load_coins(TIMEFRAME, folder)
    if candles_by_coin is None:
        print("missing 1d candle files: run python fetch_trend_data.py first")
        return
    print(f"=== short side, 1d donchian {RULE['entry']}/{RULE['exit']} mirrored, "
          f"{BASE_FEES} fees, funding {FUNDING_PER_DAY * 100:g}%/day paid ===")
    checks, trades, base = run_checks(candles_by_coin)

    print(f"{'coin':6}{'trades':>8}{'total $':>10}{'liquidated':>12}")
    for coin in COINS:
        liquidations = sum(was_liquidated(candles_by_coin[coin], t) for t in trades[coin])
        print(f"{coin:6}{len(trades[coin]):>8}{sum(base[coin]):>10.2f}{liquidations:>12}")
    print()
    for name, detail, passed in checks:
        print(f"  {'pass' if passed else 'FAIL'}  {name:13} {detail}")
    survived = all(passed for _, _, passed in checks)
    print(f"  -> {'SURVIVES' if survived else 'does not survive'} "
          f"({sum(p for _, _, p in checks)} of {len(checks)})\n")

    no_funding = sum(all_profits(coin_short_profits(trades, candles_by_coin, BASE_FEES, 0)))
    print(f"For information: same shorts with no funding cost: {no_funding:.2f}\n")

    longs = long_by_year(candles_by_coin)
    shorts = by_year(trades, base)
    print("For information: long side, short side and both together, by year:")
    print(f"{'year':6}{'long $':>10}{'short $':>10}{'both $':>10}")
    for year, l, s, both in year_rows(longs, shorts):
        print(f"{year:<6}{l:>10.2f}{s:>10.2f}{both:>10.2f}")


if __name__ == "__main__":
    main()
