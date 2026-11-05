import contextlib
import io
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import robustness
from robustness import (
    coin_profits,
    coin_trades,
    load_coins,
    neighbors_pass,
    profit_by_year,
    rule_label,
    without_best_trades,
    without_best_year,
)
from trend import COINS, candle_file

H = 3_600_000


def trade(exit_time, entry_price, exit_price):
    return {"exit_time": exit_time, "entry_price": entry_price, "exit_price": exit_price}


class TestLocked(unittest.TestCase):
    """Fixed before running. A failure here means someone changed them."""

    def test_candidates(self):
        got = [(c["name"], c["timeframe"], c["rule"], c["neighbors"])
               for c in robustness.CANDIDATES]
        self.assertEqual(got, [
            ("1d donchian 20/10", "1d", {"kind": "donchian", "entry": 20, "exit": 10}, [
                {"kind": "donchian", "entry": 15, "exit": 7},
                {"kind": "donchian", "entry": 25, "exit": 12},
                {"kind": "donchian", "entry": 30, "exit": 15},
            ]),
            ("4h ma 50/200", "4h", {"kind": "ma", "fast": 50, "slow": 200}, [
                {"kind": "ma", "fast": 40, "slow": 160},
                {"kind": "ma", "fast": 60, "slow": 240},
                {"kind": "ma", "fast": 50, "slow": 150},
            ]),
        ])

    def test_rules(self):
        self.assertEqual(robustness.BASE_FEES, "kraken taker")
        self.assertEqual(robustness.HIGH_FEES, "coinbase adv1 taker")
        self.assertEqual(robustness.LATE_DELAY, 2)
        self.assertEqual(robustness.NEIGHBOR_PROFIT_FACTOR, 1.1)
        self.assertEqual(robustness.NEIGHBOR_MIN_GOOD, 2)


class TestYears(unittest.TestCase):
    def test_profit_by_year(self):
        # no fees isn't a fee level, so use kraken maker 0.22% and work it by hand:
        # 10 -> 12: 99.78 / 10 = 9.978 coins; * 12 = 119.736; * 0.9978 = 119.4726 -> 19.4726
        trades = {"A": [trade("2023-05-01 00:00", 10, 12)],
                  "B": [trade("2023-09-01 00:00", 10, 12), trade("2024-01-01 00:00", 10, 12)]}
        years = profit_by_year(trades, "kraken maker")
        self.assertEqual(list(years), [2023, 2024])
        self.assertAlmostEqual(years[2023], 2 * 19.4726, places=3)
        self.assertAlmostEqual(years[2024], 19.4726, places=3)

    def test_sorted_by_year(self):
        trades = {"A": [trade("2025-01-01 00:00", 10, 11), trade("2022-01-01 00:00", 10, 11)]}
        self.assertEqual(list(profit_by_year(trades, "kraken taker")), [2022, 2025])

    def test_without_best_year_passes(self):
        # total 70, best year 50 -> 20 left
        self.assertEqual(without_best_year({2022: -10, 2023: 50, 2024: 30}), 20)

    def test_without_best_year_fails(self):
        # total 70, best year 80 -> -10 left
        self.assertEqual(without_best_year({2023: 80, 2024: -10}), -10)

    def test_one_year_leaves_nothing(self):
        self.assertEqual(without_best_year({2024: 50}), 0)

    def test_no_years(self):
        self.assertEqual(without_best_year({}), 0)


class TestWithoutBestTrades(unittest.TestCase):
    def test_removes_each_coins_biggest_winner(self):
        # A: 30 - 5 + 10, drop 30 -> 5.  B: only 20, drop it -> 0.  Total 5
        self.assertEqual(without_best_trades({"A": [30, -5, 10], "B": [20]}), 5)

    def test_can_turn_negative(self):
        # A: drop 30 -> -5
        self.assertEqual(without_best_trades({"A": [30, -5]}), -5)

    def test_coin_without_winners_keeps_its_losses(self):
        # B has no winner, so its -2 stays: 5 - 2 = 3
        self.assertEqual(without_best_trades({"A": [30, -5, 10], "B": [-2]}), 3)

    def test_coin_with_no_trades(self):
        self.assertEqual(without_best_trades({"A": [], "B": [4, 6]}), 4)


class TestNeighbors(unittest.TestCase):
    # profit factors: [3, -1] = 3.0, [11, -10] = 1.1, [2, -2] = 1.0

    def test_all_positive_two_good(self):
        self.assertTrue(neighbors_pass([[3, -1], [3, -1], [11, -10]]))

    def test_one_negative_fails(self):
        self.assertFalse(neighbors_pass([[3, -1], [3, -1], [1, -2]]))

    def test_only_one_above_1_1_fails(self):
        # [11, -10] is exactly 1.1, not above
        self.assertFalse(neighbors_pass([[3, -1], [11, -10], [11, -10]]))

    def test_zero_total_is_not_positive(self):
        self.assertFalse(neighbors_pass([[3, -1], [3, -1], [2, -2]]))


class TestLabels(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(rule_label({"kind": "donchian", "entry": 15, "exit": 7}), "donchian 15/7")
        self.assertEqual(rule_label({"kind": "ma", "fast": 40, "slow": 160}), "ma 40/160")


def candle_dicts(pairs, step_hours=24):
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    out = []
    for i, (o, c) in enumerate(pairs):
        moment = start + timedelta(hours=i * step_hours)
        out.append({"timestamp": int(moment.timestamp() * 1000),
                    "date": moment.strftime("%Y-%m-%d %H:%M"), "open": o, "close": c})
    return out


class TestCoinTradesAndProfits(unittest.TestCase):
    # donchian 3/2 (as in test_trend): buy at h4 open 13.5, sell at h6 open 11.5
    PAIRS = [(10, 10), (10, 11), (11, 12), (12, 13), (13.5, 14), (14, 12), (11.5, 11.5), (11, 11)]
    RULE = {"kind": "donchian", "entry": 3, "exit": 2}

    def test_one_trade_per_coin(self):
        trades = coin_trades({"A": candle_dicts(self.PAIRS)}, self.RULE)
        self.assertEqual([(t["entry_price"], t["exit_price"]) for t in trades["A"]], [(13.5, 11.5)])

    def test_late_uses_delay(self):
        # same signals filled 2 candles later: buy at i5 open 14, sell at i7 open 11
        trades = coin_trades({"A": candle_dicts(self.PAIRS)}, self.RULE, delay=2)
        self.assertEqual([(t["entry_price"], t["exit_price"]) for t in trades["A"]], [(14, 11)])

    def test_open_trade_counted(self):
        # stop after i5 with close 15 instead: still holding, counted at last close 15
        pairs = self.PAIRS[:5] + [(14, 15)]
        trades = coin_trades({"A": candle_dicts(pairs)}, self.RULE)
        self.assertEqual(trades["A"][0]["exit_price"], 15)
        self.assertEqual(trades["A"][0]["reason"], "still open")

    def test_coin_profits_uses_fee_level(self):
        # 10 -> 12 at coinbase adv1 taker 0.75%:
        # 99.25 / 10 = 9.925 coins; * 12 = 119.1; * 0.9925 = 118.20675 -> 18.20675
        profits = coin_profits({"A": [trade("2023-01-01 00:00", 10, 12)]}, "coinbase adv1 taker")
        self.assertAlmostEqual(profits["A"][0], 18.20675)


class TestMain(unittest.TestCase):
    def write_files(self, folder, timeframes, coins=COINS):
        # price rises steadily from 10 to 20 over 300 candles, starting 2023-01-01
        closes = [10 + i * 10 / 299 for i in range(300)]
        for tf in timeframes:
            step = 24 if tf == "1d" else 4
            candles = candle_dicts([(10 if i == 0 else closes[i - 1], c)
                                    for i, c in enumerate(closes)], step)
            for coin in coins:
                with open(os.path.join(folder, candle_file(coin, tf)), "w") as f:
                    f.write("timestamp,date,open,high,low,close,volume\n")
                    for c in candles:
                        f.write(f"{c['timestamp']},{c['date']},{c['open']},{c['close']},"
                                f"{c['open']},{c['close']},1\n")

    def run_main(self, timeframes, coins=COINS):
        with tempfile.TemporaryDirectory() as folder:
            self.write_files(folder, timeframes, coins)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                robustness.main(folder)
        return out.getvalue()

    def test_load_coins_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_files(folder, ["1d"], COINS[:3])
            self.assertIsNone(load_coins("1d", folder))

    def test_missing_files_message(self):
        out = self.run_main(["1d"])
        self.assertIn("missing 4h candle files", out)
        self.assertNotIn("missing 1d", out)

    def test_steady_rise(self):
        # a steady rise: one trade per coin, still open at the end, all in one
        # year and all winners. So:
        #   1 years      FAIL (one year: nothing left without it)
        #   2 best trade FAIL (each coin's only trade is its best)
        #   3 neighbors  pass (every neighbor wins once, no losses: PF inf)
        #   4 high fees  pass (10 -> 20 beats any fee)
        #   5 late       pass
        out = self.run_main(["1d", "4h"])
        for name in ("1d donchian 20/10", "4h ma 50/200"):
            section = out.split(f"=== {name}")[1].split("\n=== ")[0]
            self.assertIn("baseline: 6 trades", section)
            self.assertIn("FAIL  1 years", section)
            self.assertIn("FAIL  2 best trade", section)
            self.assertIn("pass  3 neighbors", section)
            self.assertIn("pass  4 high fees", section)
            self.assertIn("pass  5 late", section)
            self.assertIn("does not survive (3 of 5 checks)", section)

    def test_neighbor_totals(self):
        # Donchian 15/7 on the steady rise: the first close above the previous 15
        # is i15, so it buys at i16's open = close 15 = 10 + 15 * 10 / 299, and is
        # still holding at the last close, 20. Kraken taker 0.38% each side, and
        # six identical coins:
        entry = 10 + 15 * 10 / 299
        one_coin = 100 * (1 - 0.0038) / entry * 20 * (1 - 0.0038) - 100
        out = self.run_main(["1d", "4h"])
        self.assertIn(f"donchian 15/7: {6 * one_coin:.2f}", out)
        # MA 40/160: the trend is known (and up) from i159, so it buys at i160's
        # open = close 159
        entry = 10 + 159 * 10 / 299
        one_coin = 100 * (1 - 0.0038) / entry * 20 * (1 - 0.0038) - 100
        self.assertIn(f"ma 40/160: {6 * one_coin:.2f}", out)


if __name__ == "__main__":
    unittest.main()
