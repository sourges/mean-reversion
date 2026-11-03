import contextlib
import io
import os
import tempfile
import unittest

import holdout
from holdout import evaluate, main, max_drawdown, passes, profit_factor, run_setting

H = 3_600_000


def make_candles(pairs):
    return [{"timestamp": h * H, "date": f"h{h}", "open": o, "close": c}
            for h, (o, c) in enumerate(pairs)]


# window 3, entry -1: buys at h3 open 8, sells at h5 open 11 (see test_backtest)
BASIC = [(10, 10), (10, 10), (10, 7), (8, 8), (9, 10), (11, 11)]


def setting(**changes):
    s = {"name": "t", "window": 3, "entry_z": -1, "exit_z": 0.0, "trend_window": None}
    s.update(changes)
    return s


class TestLockedSettings(unittest.TestCase):
    """The list and rule were fixed before seeing the unseen data.
    If one of these fails, something changed that shouldn't have."""

    def test_six_settings_exactly(self):
        got = [(s["window"], s["entry_z"], s["exit_z"], s["trend_window"])
               for s in holdout.SETTINGS]
        self.assertEqual(got, [
            (20, -2.0, 0.0, None),
            (20, -2.0, 0.0, 200),
            (50, -2.0, 0.0, 200),
            (100, -2.5, 0.0, None),
            (100, -2.5, 0.0, 200),
            (20, -2.0, 0.5, 200),
        ])

    def test_pass_rule(self):
        self.assertEqual(holdout.PASS_FEES, "kraken taker")
        self.assertEqual(holdout.PASS_PROFIT_FACTOR, 1.1)


class TestProfitFactor(unittest.TestCase):
    def test_mixed(self):
        # wins 2 + 3 = 5, losses 1 + 1 = 2
        self.assertEqual(profit_factor([2, -1, 3, -1]), 2.5)

    def test_no_losses(self):
        self.assertEqual(profit_factor([2, 3]), float("inf"))

    def test_only_losses(self):
        self.assertEqual(profit_factor([-1, -2]), 0.0)

    def test_no_trades(self):
        self.assertEqual(profit_factor([]), 0.0)


class TestMaxDrawdown(unittest.TestCase):
    def test_mixed(self):
        # running profit 2, 1, 4, 3: worst drop from a high is 1
        self.assertEqual(max_drawdown([2, -1, 3, -1]), 1)

    def test_losing_from_the_start(self):
        # running profit -1, -3 below the starting 0
        self.assertEqual(max_drawdown([-1, -2]), 3)

    def test_drop_below_zero_after_a_high(self):
        # running profit 5, 2, 3, -1: from the high of 5 down to -1
        self.assertEqual(max_drawdown([5, -3, 1, -4]), 6)

    def test_only_wins(self):
        self.assertEqual(max_drawdown([1, 2, 3]), 0)

    def test_no_trades(self):
        self.assertEqual(max_drawdown([]), 0)


class TestPasses(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(passes({"profit_factor": 1.2, "total": 5}))

    def test_exactly_1_1_is_not_above(self):
        self.assertFalse(passes({"profit_factor": 1.1, "total": 5}))

    def test_needs_positive_total(self):
        self.assertFalse(passes({"profit_factor": 2.0, "total": 0}))

    def test_no_losses_passes(self):
        self.assertTrue(passes({"profit_factor": float("inf"), "total": 3}))


class TestEvaluate(unittest.TestCase):
    def test_no_fees(self):
        # profits 37.5 and -10
        trades = [{"entry_price": 8, "exit_price": 11}, {"entry_price": 10, "exit_price": 9}]
        self.assertEqual(evaluate(trades, 0, 0),
                         {"trades": 2, "total": 27.5, "profit_factor": 3.75, "drawdown": 10})

    def test_uses_fees(self):
        # 1% each side: 99 / 8 * 11 * 0.99 - 100 = 34.76375
        r = evaluate([{"entry_price": 8, "exit_price": 11}], 0.01, 0.01)
        self.assertAlmostEqual(r["total"], 34.76375)


class TestRunSetting(unittest.TestCase):
    def test_basic(self):
        trades = run_setting(make_candles(BASIC), setting())
        self.assertEqual([(t["entry_price"], t["exit_price"]) for t in trades], [(8, 11)])

    def test_uses_trend_window(self):
        # 5-candle trend isn't known by the h2 dip, so no buy
        self.assertEqual(run_setting(make_candles(BASIC), setting(trend_window=5)), [])

    def test_uses_exit_z(self):
        # the h4 z is +1.34, short of 2, so the trade never closes
        self.assertEqual(run_setting(make_candles(BASIC), setting(exit_z=2.0)), [])

    def test_uses_entry_z(self):
        # the dip's z is -1.414, not below -1.5
        self.assertEqual(run_setting(make_candles(BASIC), setting(entry_z=-1.5)), [])


class TestMain(unittest.TestCase):
    # same 30 candles as test_backtest's TestMain: with window 20 and z -2,
    # one trade, 8 -> 10.5; Kraken taker profit 30.2544, no losses -> PF inf.
    # Every other setting needs 100 or 200 candles of history: no trades.
    PAIRS = [(10, 10 + (i % 2) * 0.1) for i in range(25)]
    PAIRS += [(10, 8), (8, 8.5), (8.5, 10.5), (10.5, 10.5), (10.5, 10.5)]

    def run_main(self, args):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "candles_test_1h.csv")
            with open(path, "w", newline="") as f:
                f.write("timestamp,date,open,high,low,close,volume\n")
                for h, (o, c) in enumerate(self.PAIRS):
                    f.write(f"{h * H},h{h},{o},{o},{c},{c},1\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main([path] if args is None else args)
        return out.getvalue()

    def rows(self, out):
        return {line[:20].strip(): line for line in out.splitlines()
                if line[:1].isdigit()}

    def test_one_row_per_setting(self):
        self.assertEqual(len(self.rows(self.run_main(None))), 6)

    def test_baseline_trades_and_passes(self):
        row = self.rows(self.run_main(None))["1 baseline"].split()
        # name is two words, then: trades, kraken total/PF/DD, adv2 total/PF/DD, result
        self.assertEqual(row[2], "1")
        self.assertEqual(row[3], "30.25")
        self.assertEqual(row[4], "inf")
        self.assertEqual(row[-1], "PASS")

    def test_settings_without_trades_fail(self):
        rows = self.rows(self.run_main(None))
        for name, line in rows.items():
            if name != "1 baseline":
                self.assertTrue(line.endswith("fail"), name)

    def test_buy_and_hold_line(self):
        # 99.62 / 10 = 9.962 coins; * 10.5 = 104.601; * 0.9962 = 104.2035 -> 4.20
        self.assertIn("Buy and hold, $100 at kraken taker fees: 4.20", self.run_main(None))

    def test_verdict_uses_kraken_taker_fees(self):
        # same dip, but the sell fills at 8.04: +0.5% before fees.
        # 99.62 / 8 = 12.4525 coins; * 8.04 = 100.1181; * 0.9962 = 99.7377
        #   -> -0.26 at Kraken taker: a loss, so it must fail
        self.PAIRS = self.PAIRS[:28] + [(8.04, 8.04), (8.04, 8.04)]
        row = self.rows(self.run_main(None))["1 baseline"].split()
        self.assertEqual(row[3], "-0.26")
        self.assertEqual(row[-1], "fail")

    def test_usage_without_file(self):
        self.assertIn("Usage", self.run_main([]))


if __name__ == "__main__":
    unittest.main()
