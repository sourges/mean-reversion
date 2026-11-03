import contextlib
import csv
import io
import math
import os
import tempfile
import unittest

from backtest import (
    buy_and_hold,
    exit_reason,
    find_trades,
    main,
    read_candles,
    summarize,
    trade_profit,
    write_trades,
    zscore,
)

H = 3_600_000  # one hour in milliseconds


def make_candles(pairs, hours=None):
    """[(open, close), ...] -> candle dicts at hours 0, 1, 2 ... (or `hours`)."""
    hours = hours or list(range(len(pairs)))
    return [{"timestamp": h * H, "date": f"h{h}", "open": o, "close": c}
            for h, (o, c) in zip(hours, pairs)]


# Hand-worked, window 3, entry z < -1, exit z >= 0:
#   i2 closes [10, 10, 7]: mean 9, std sqrt(2) = 1.414, z = -2/1.414 = -1.414 -> buy
#      fill at i3 open = 8 (not the 7 close that triggered it)
#   i3 closes [10, 7, 8]: mean 8.333, z = -0.27 -> hold
#   i4 closes [7, 8, 10]: mean 8.333, z = +1.34 -> sell
#      fill at i5 open = 11 (not the 10 close)
BASIC = [(10, 10), (10, 10), (10, 7), (8, 8), (9, 10), (11, 11)]


class TestZscore(unittest.TestCase):
    def test_not_enough_candles(self):
        self.assertIsNone(zscore([10, 10], 1, 3))

    def test_flat_prices_have_no_z(self):
        self.assertIsNone(zscore([10, 10, 10], 2, 3))

    def test_dip(self):
        self.assertAlmostEqual(zscore([10, 10, 7], 2, 3), -math.sqrt(2))

    def test_at_the_average(self):
        # mean of 9, 11, 10 is 10
        self.assertEqual(zscore([9, 11, 10], 2, 3), 0)

    def test_only_uses_window_ending_at_i(self):
        # the 100 before and the 50 after are ignored: same as [10, 10, 7]
        self.assertAlmostEqual(zscore([100, 10, 10, 7, 50], 3, 3), -math.sqrt(2))


class TestExitReason(unittest.TestCase):
    def test_back_above_average(self):
        self.assertEqual(exit_reason(0.5, 9, 8, H, 0, None, None), "signal")

    def test_exactly_at_average(self):
        self.assertEqual(exit_reason(0.0, 9, 8, H, 0, None, None), "signal")

    def test_still_below_average(self):
        self.assertIsNone(exit_reason(-0.1, 9, 8, H, 0, None, None))

    def test_no_z_keeps_holding(self):
        self.assertIsNone(exit_reason(None, 9, 8, H, 0, None, None))

    def test_price_stop_exactly_at_level(self):
        # 25% below 8 is 6
        self.assertEqual(exit_reason(-1, 6.0, 8, H, 0, 0.25, None), "stop")

    def test_price_stop_just_above_level(self):
        self.assertIsNone(exit_reason(-1, 6.01, 8, H, 0, 0.25, None))

    def test_time_stop_reached(self):
        self.assertEqual(exit_reason(-1, 9, 8, 2 * H, 0, None, 2), "time")

    def test_time_stop_not_yet(self):
        self.assertIsNone(exit_reason(-1, 9, 8, 1 * H, 0, None, 2))

    def test_signal_beats_stop(self):
        self.assertEqual(exit_reason(0.5, 5, 8, H, 0, 0.25, None), "signal")

    def test_stop_beats_time(self):
        self.assertEqual(exit_reason(-1, 5, 8, 5 * H, 0, 0.25, 2), "stop")


class TestFindTrades(unittest.TestCase):
    def test_basic_trade_fills_at_next_open(self):
        trades, open_trade = find_trades(make_candles(BASIC), 3, -1)
        self.assertEqual(trades, [{
            "entry_time": "h3", "entry_price": 8,
            "exit_time": "h5", "exit_price": 11,
            "reason": "signal", "hours": 2,
        }])
        self.assertIsNone(open_trade)

    def test_entry_needs_z_below_entry_level(self):
        # the dip's z is -1.414, not below -1.5
        trades, open_trade = find_trades(make_candles(BASIC), 3, -1.5)
        self.assertEqual(trades, [])
        self.assertIsNone(open_trade)

    def test_signal_on_last_candle_cannot_fill(self):
        trades, open_trade = find_trades(make_candles(BASIC[:3]), 3, -1)
        self.assertEqual(trades, [])
        self.assertIsNone(open_trade)

    def test_still_holding_at_the_end(self):
        trades, open_trade = find_trades(make_candles(BASIC[:4]), 3, -1)
        self.assertEqual(trades, [])
        self.assertEqual(open_trade["entry_time"], "h3")
        self.assertEqual(open_trade["entry_price"], 8)

    def test_sells_when_z_is_exactly_zero(self):
        # buy at 8 (h3); i3 [10, 7, 6] -> hold
        # i4 [7, 6, 6.5]: mean 6.5 = close, z = 0 -> sell at i5 open 6.4
        candles = make_candles([(10, 10), (10, 10), (10, 7), (8, 6),
                                (6.5, 6.5), (6.4, 6.4), (6.3, 6.3)])
        trades, _ = find_trades(candles, 3, -1)
        self.assertEqual(trades[0]["exit_price"], 6.4)
        self.assertEqual(trades[0]["hours"], 2)

    def test_buys_again_after_selling(self):
        # BASIC, then [11, 11, 8] at i7 -> buy again at i8 open 9
        candles = make_candles(BASIC + [(11, 11), (11, 8), (9, 9)])
        trades, open_trade = find_trades(candles, 3, -1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(open_trade["entry_time"], "h8")
        self.assertEqual(open_trade["entry_price"], 9)


class TestStops(unittest.TestCase):
    # buy at 8 (h3); closes 7, 6 keep z negative
    # i4 close 6 is exactly 25% below 8 -> stop, sell at i5 open 5.9
    FALLING = [(10, 10), (10, 10), (10, 7), (8, 7), (7, 6), (5.9, 5.9), (5.8, 5.8)]

    # buy at 8 (h3); closes 6.9, 6.8 keep z negative, never 25% down
    DRIFTING = [(10, 10), (10, 10), (10, 7), (8, 6.9), (6.8, 6.8), (6.7, 6.7)]

    def test_price_stop(self):
        trades, _ = find_trades(make_candles(self.FALLING), 3, -1, stop_pct=0.25)
        self.assertEqual(trades[0]["reason"], "stop")
        self.assertEqual(trades[0]["exit_time"], "h5")
        self.assertEqual(trades[0]["exit_price"], 5.9)

    def test_no_price_stop_keeps_holding(self):
        trades, open_trade = find_trades(make_candles(self.FALLING), 3, -1)
        self.assertEqual(trades, [])
        self.assertEqual(open_trade["entry_price"], 8)

    def test_time_stop(self):
        # held 1h at i3 close, 2h at i4 close -> sell at i5 open 6.7
        trades, _ = find_trades(make_candles(self.DRIFTING), 3, -1, max_hours=2)
        self.assertEqual(trades[0]["reason"], "time")
        self.assertEqual(trades[0]["exit_price"], 6.7)
        self.assertEqual(trades[0]["hours"], 2)

    def test_time_stop_one_hour(self):
        trades, _ = find_trades(make_candles(self.DRIFTING), 3, -1, max_hours=1)
        self.assertEqual(trades[0]["exit_time"], "h4")
        self.assertEqual(trades[0]["hours"], 1)

    def test_time_stop_counts_clock_hours_across_a_gap(self):
        # exchange outage: candle after h3 is h10. At its close (11h) the
        # trade has been held 8 hours, so a 5h stop fires: sell at h11.
        candles = make_candles(self.DRIFTING, hours=[0, 1, 2, 3, 10, 11])
        trades, _ = find_trades(candles, 3, -1, max_hours=5)
        self.assertEqual(trades[0]["exit_time"], "h11")
        self.assertEqual(trades[0]["hours"], 8)


class TestProfit(unittest.TestCase):
    def test_no_fees(self):
        # $100 at 8 = 12.5 coins; at 11 = $137.50
        self.assertAlmostEqual(trade_profit(8, 11, 0, 0), 37.5)

    def test_one_percent_each_side(self):
        # 99 / 8 = 12.375 coins; 12.375 * 11 = 136.125; * 0.99 = 134.76375
        self.assertAlmostEqual(trade_profit(8, 11, 0.01, 0.01), 34.76375)

    def test_loss(self):
        self.assertAlmostEqual(trade_profit(10, 9, 0, 0), -10)

    def test_flat_price_loses_the_fees(self):
        # 99.62 / 10 = 9.962 coins; * 10 * 0.9962 = 99.241444
        self.assertAlmostEqual(trade_profit(10, 10, 0.0038, 0.0038), -0.758556)

    def test_other_trade_size(self):
        self.assertAlmostEqual(trade_profit(8, 11, 0, 0, usd=500), 187.5)


class TestSummary(unittest.TestCase):
    def test_summary(self):
        # profits 37.5, -10, 0 -> one win (0 isn't a win)
        trades = [
            {"entry_price": 8, "exit_price": 11},
            {"entry_price": 10, "exit_price": 9},
            {"entry_price": 10, "exit_price": 10},
        ]
        s = summarize(trades, 0, 0)
        self.assertEqual(s["trades"], 3)
        self.assertAlmostEqual(s["win_rate"], 33.333333, places=4)
        self.assertAlmostEqual(s["total"], 27.5)
        self.assertAlmostEqual(s["average"], 27.5 / 3)

    def test_summary_uses_fees(self):
        s = summarize([{"entry_price": 8, "exit_price": 11}], 0.01, 0.01)
        self.assertAlmostEqual(s["total"], 34.76375)

    def test_no_trades(self):
        self.assertEqual(summarize([], 0, 0),
                         {"trades": 0, "win_rate": 0.0, "total": 0.0, "average": 0.0})

    def test_buy_and_hold_first_open_to_last_close(self):
        candles = make_candles([(10, 12), (12, 8), (8, 5)])
        self.assertAlmostEqual(buy_and_hold(candles, 0, 0), -50)


class TestFiles(unittest.TestCase):
    def test_read_candles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "c.csv")
            with open(path, "w", newline="") as f:
                f.write("timestamp,date,open,high,low,close,volume\n")
                f.write("1696118400000,2023-10-01 00:00,0.5178,0.5209,0.5178,0.5205,2864.7\n")
            candles = read_candles(path)
        self.assertEqual(candles, [{"timestamp": 1696118400000, "date": "2023-10-01 00:00",
                                    "open": 0.5178, "close": 0.5205}])

    def test_write_trades_has_profit_column_for_report(self):
        trades, _ = find_trades(make_candles(BASIC), 3, -1)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "t.csv")
            write_trades(trades, path, 0.0, 0.0)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(rows, [{
            "entry_time": "h3", "entry_price": "8", "exit_time": "h5",
            "exit_price": "11", "reason": "signal", "hours": "2",
            "profit": "37.5000",
        }])
        # this is exactly how trade-analyzer's read_profits reads it
        self.assertEqual(float(rows[0]["profit"]), 37.5)

    def test_write_trades_uses_fees(self):
        # buy at 10, sell at 10 with 0.38% each side: -0.758556
        trades = [{"entry_time": "a", "entry_price": 10, "exit_time": "b",
                   "exit_price": 10, "reason": "time", "hours": 48}]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "t.csv")
            write_trades(trades, path, 0.0038, 0.0038)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["profit"], "-0.7586")

    def test_main_runs_and_saves_trades(self):
        # 25 flat-ish candles so the default window of 20 fills,
        # then a dip and a recovery
        pairs = [(10, 10 + (i % 2) * 0.1) for i in range(25)]
        pairs += [(10, 8), (8, 8.5), (8.5, 10.5), (10.5, 10.5), (10.5, 10.5)]
        with tempfile.TemporaryDirectory() as folder:
            old = os.getcwd()
            os.chdir(folder)
            try:
                with open("c.csv", "w", newline="") as f:
                    f.write("timestamp,date,open,high,low,close,volume\n")
                    for h, (o, c) in enumerate(pairs):
                        f.write(f"{h * H},h{h},{o},{o},{c},{c},1\n")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    main(["c.csv"])
                with open("backtest_trades.csv", newline="") as f:
                    rows = list(csv.DictReader(f))
            finally:
                os.chdir(old)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entry_time"], "h26")
        self.assertIn("Saved 1 trades", out.getvalue())


if __name__ == "__main__":
    unittest.main()
