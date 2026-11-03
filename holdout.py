"""Run the fixed list of settings ONCE on candles the strategy was never tuned on.

Usage (mean-reversion folder, venv active):
    python holdout.py candles_coinbase_crv_usd_1h_2021-10-01_to_2023-10-01.csv

The list and the pass rule below were chosen on Nov 3, 2026, BEFORE looking
at any 2021-2023 candles. Don't change them after seeing the results: that
would turn this unseen data into tuning data, and the test would stop
meaning anything.
"""

import sys

from backtest import FEES, buy_and_hold, find_trades, read_candles, trade_profit

# --- fixed in advance (Nov 3, 2026) ----------------------------------------
SETTINGS = [
    {"name": "1 baseline", "window": 20, "entry_z": -2.0, "exit_z": 0.0, "trend_window": None},
    {"name": "2 trend filter", "window": 20, "entry_z": -2.0, "exit_z": 0.0, "trend_window": 200},
    {"name": "3 slower average", "window": 50, "entry_z": -2.0, "exit_z": 0.0, "trend_window": 200},
    {"name": "4 best from tuning", "window": 100, "entry_z": -2.5, "exit_z": 0.0, "trend_window": None},
    {"name": "5 best + filter", "window": 100, "entry_z": -2.5, "exit_z": 0.0, "trend_window": 200},
    {"name": "6 hold past middle", "window": 20, "entry_z": -2.0, "exit_z": 0.5, "trend_window": 200},
]
PASS_FEES = "kraken taker"  # the pass rule is judged at these fees
PASS_PROFIT_FACTOR = 1.1  # must be ABOVE this, and the total must be positive
SHOW_FEES = ["kraken taker", "coinbase adv2 taker"]
# ---------------------------------------------------------------------------


def profit_factor(profits):
    """Total wins / total losses. inf if nothing lost, 0.0 if nothing traded."""
    wins = sum(p for p in profits if p > 0)
    losses = -sum(p for p in profits if p < 0)
    if not profits:
        return 0.0
    if losses == 0:
        return float("inf")
    return wins / losses


def max_drawdown(profits):
    """Biggest drop in running profit from a high point (running profit starts at 0)."""
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for p in profits:
        equity += p
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def evaluate(trades, buy_fee, sell_fee):
    """Count, total, profit factor and max drawdown at one fee level."""
    profits = [trade_profit(t["entry_price"], t["exit_price"], buy_fee, sell_fee)
               for t in trades]
    return {
        "trades": len(profits),
        "total": sum(profits),
        "profit_factor": profit_factor(profits),
        "drawdown": max_drawdown(profits),
    }


def passes(result):
    """The rule fixed in advance: profit factor above 1.1 AND a positive total."""
    return result["profit_factor"] > PASS_PROFIT_FACTOR and result["total"] > 0


def run_setting(candles, setting):
    """Trades for one entry in SETTINGS (no stops in this test)."""
    trades, _ = find_trades(candles, setting["window"], setting["entry_z"],
                            setting["exit_z"], trend_window=setting["trend_window"])
    return trades


def main(args):
    if not args:
        print("Usage: python holdout.py <candle file>")
        return
    path = args[0]
    candles = read_candles(path)
    print(f"{path}: {len(candles)} candles, {candles[0]['date']} to {candles[-1]['date']} UTC")
    print(f"Pass rule: profit factor > {PASS_PROFIT_FACTOR} and total > 0 at {PASS_FEES} fees\n")

    groups = f"{'':20}{'':>7}"
    columns = f"{'setting':20}{'trades':>7}"
    for fee_name in SHOW_FEES:
        groups += f"  {fee_name:^23}"
        columns += f"  {'total $':>9}{'PF':>6}{'DD $':>8}"
    print(groups)
    print(columns + "  result")

    for setting in SETTINGS:
        trades = run_setting(candles, setting)
        line = f"{setting['name']:20}{len(trades):>7}"
        for fee_name in SHOW_FEES:
            r = evaluate(trades, *FEES[fee_name])
            line += f"  {r['total']:>9.2f}{r['profit_factor']:>6.2f}{r['drawdown']:>8.2f}"
        verdict = "PASS" if passes(evaluate(trades, *FEES[PASS_FEES])) else "fail"
        print(line + f"  {verdict}")

    hold = buy_and_hold(candles, *FEES[PASS_FEES])
    print(f"\nBuy and hold, $100 at {PASS_FEES} fees: {hold:.2f}")
    print("PF = profit factor, DD = max drawdown. One run, no changes afterward.")


if __name__ == "__main__":
    main(sys.argv[1:])
