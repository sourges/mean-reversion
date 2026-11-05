import contextlib
import io
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import shorts
from shorts import (
    by_year,
    coin_short_profits,
    coin_short_trades,
    passes,
    short_entry,
    short_exit,
    short_profit,
    was_liquidated,
    year_rows,
)
from trend import COINS, candle_file

DAY_MS = 86_400_000


def daily(pairs):
    """[(open, close), ...] -> daily candle dicts from 2023-01-01."""
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    out = []
    for i, (o, c) in enumerate(pairs):
        moment = start + timedelta(days=i)
        out.append({"timestamp": int(moment.timestamp() * 1000),
                    "date": moment.strftime("%Y-%m-%d %H:%M"), "open": o, "close": c})
    return out


class TestLocked(unittest.TestCase):
    """Fixed before running. A failure here means someone changed them."""

    def test_settings(self):
        self.assertEqual(shorts.TIMEFRAME, "1d")
        self.assertEqual(shorts.RULE, {"entry": 20, "exit": 10})
        self.assertEqual(shorts.NEIGHBORS, [{"entry": 15, "exit": 7}, {"entry": 25, "exit": 12},
                                            {"entry": 30, "exit": 15}])
        self.assertEqual(shorts.BASE_FEES, "kraken taker")
        self.assertEqual(shorts.HIGH_FEES, "coinbase adv1 taker")
        self.assertEqual(shorts.FUNDING_PER_DAY, 0.0003)
        self.assertEqual(shorts.LATE_DELAY, 2)
        self.assertEqual(shorts.PASS_MIN_TRADES, 30)
        self.assertEqual(shorts.PASS_PROFIT_FACTOR, 1.1)
        self.assertEqual(shorts.USD, 100.0)


class TestRule(unittest.TestCase):
    def test_new_low_opens(self):
        # previous 3 closes 5, 4, 3: lowest 3, close 2 is below
        self.assertTrue(short_entry([5, 4, 3, 2], 3, 3))

    def test_equal_to_low_does_not_open(self):
        self.assertFalse(short_entry([5, 3, 4, 3], 3, 3))

    def test_oldest_low_counts(self):
        # previous 3 closes 1, 5, 4: lowest is the oldest, 1
        self.assertFalse(short_entry([1, 5, 4, 3], 3, 3))

    def test_open_needs_history(self):
        self.assertFalse(short_entry([5, 4, 0], 2, 3))

    def test_new_high_covers(self):
        # previous 3 closes 1, 2, 0: highest 2, close 3 is above
        self.assertTrue(short_exit([1, 2, 0, 3], 3, 3))

    def test_equal_to_high_does_not_cover(self):
        self.assertFalse(short_exit([1, 3, 2, 3], 3, 3))

    def test_oldest_high_counts(self):
        self.assertFalse(short_exit([5, 1, 2, 4], 3, 3))

    def test_cover_needs_history(self):
        self.assertFalse(short_exit([1, 9], 1, 3))


class TestProfit(unittest.TestCase):
    def test_win_with_all_costs(self):
        # short $100 at 10, cover at 8: gain 20
        # fees: 1% of 100 = 1, and 1% of the 80 it's worth at cover = 0.8
        # funding: 0.1% a day for 2 days = 0.2  ->  20 - 1.8 - 0.2 = 18
        self.assertAlmostEqual(short_profit(10, 8, 0.01, 0.01, 48, 0.001), 18.0)

    def test_loss(self):
        # price rose from 10 to 12: -20
        self.assertAlmostEqual(short_profit(10, 12, 0, 0, 0, 0), -20)

    def test_cover_fee_on_value_at_exit(self):
        # 10 -> 5: gain 50, cover fee 10% of the 50 it's worth = 5
        self.assertAlmostEqual(short_profit(10, 5, 0, 0.1, 0, 0), 45)

    def test_liquidated(self):
        self.assertEqual(short_profit(10, 8, 0.01, 0.01, 48, 0.001, liquidated=True), -100)


class TestLiquidation(unittest.TestCase):
    def trade(self, candles, entry, exit_price, exit_index):
        return {"entry_ts": candles[0]["timestamp"], "exit_ts": candles[exit_index]["timestamp"],
                "entry_price": entry, "exit_price": exit_price}

    def test_close_reaches_double(self):
        candles = daily([(10, 10), (10, 15), (15, 21), (12, 12)])
        self.assertTrue(was_liquidated(candles, self.trade(candles, 10, 12, 3)))

    def test_exactly_double(self):
        candles = daily([(10, 10), (10, 20), (15, 15), (12, 12)])
        self.assertTrue(was_liquidated(candles, self.trade(candles, 10, 12, 3)))

    def test_below_double(self):
        candles = daily([(10, 10), (10, 19.9), (15, 15), (12, 12)])
        self.assertFalse(was_liquidated(candles, self.trade(candles, 10, 12, 3)))

    def test_after_cover_does_not_count(self):
        # the 21 close is on the cover candle itself, after the short is closed at its open
        candles = daily([(10, 10), (10, 15), (15, 12), (12, 21)])
        self.assertFalse(was_liquidated(candles, self.trade(candles, 10, 12, 3)))

    def test_cover_price_double(self):
        candles = daily([(10, 10), (10, 15), (15, 18), (20, 20)])
        self.assertTrue(was_liquidated(candles, self.trade(candles, 10, 20, 3)))


# Short 3/2 on falling prices (the long test's candles upside down):
#   d3 close 17 < lowest of [20, 19, 18]  -> short at d4 open 16.5
#   d4 close 16, highest of [18, 17] is 18 -> hold
#   d5 close 18 > highest of [17, 16]      -> cover at d6 open 18.5
FALLING = [(20, 20), (20, 19), (19, 18), (18, 17), (16.5, 16), (16, 18), (18.5, 18.5), (19, 19)]
RULE = {"entry": 3, "exit": 2}


class TestTrades(unittest.TestCase):
    def test_short_trade(self):
        trades = coin_short_trades({"A": daily(FALLING)}, RULE)
        t = trades["A"][0]
        self.assertEqual((t["entry_time"][:10], t["entry_price"], t["exit_time"][:10],
                          t["exit_price"], t["hours"]),
                         ("2023-01-05", 16.5, "2023-01-07", 18.5, 48))

    def test_one_candle_late(self):
        # signals at d3 and d5 fill at d5 open 16 and d7 open 19
        t = coin_short_trades({"A": daily(FALLING)}, RULE, delay=2)["A"][0]
        self.assertEqual((t["entry_price"], t["exit_price"]), (16, 19))

    def test_open_short_counted_at_last_close(self):
        t = coin_short_trades({"A": daily(FALLING[:5])}, RULE)["A"][0]
        self.assertEqual((t["exit_price"], t["reason"]), (16, "still open"))

    def test_profits_no_costs(self):
        trades = coin_short_trades({"A": daily(FALLING)}, RULE)
        profits = coin_short_profits(trades, {"A": daily(FALLING)}, "kraken taker", 0)
        # 16.5 -> 18.5 is a loss: 100 * -2 / 16.5 = -12.12, then fees:
        # 0.38 + 0.38% of 100 * 18.5 / 16.5 (112.12) = 0.38 + 0.426 = 0.806
        expected = 100 * (16.5 - 18.5) / 16.5 - 0.38 - 100 * 18.5 / 16.5 * 0.0038
        self.assertAlmostEqual(profits["A"][0], expected)

    def test_profits_with_funding(self):
        # 10 -> 8 held 48 h at Kraken taker and 0.03%/day:
        # gain 20; fees 0.38 + 0.304 = 0.684; funding 0.06  ->  19.256
        candles = daily([(10, 10), (10, 9), (8, 8)])
        trades = {"A": [{"entry_ts": candles[0]["timestamp"], "exit_ts": candles[2]["timestamp"],
                         "entry_price": 10, "exit_price": 8, "hours": 48,
                         "exit_time": "2023-01-03 00:00"}]}
        profits = coin_short_profits(trades, {"A": candles}, "kraken taker", 0.0003)
        self.assertAlmostEqual(profits["A"][0], 19.256)

    def test_profits_use_the_fee_level(self):
        # same 10 -> 8 at Coinbase Adv 1 taker 0.75%, no funding:
        # gain 20; fees 0.75 + 0.75% of 80 = 0.6  ->  18.65
        candles = daily([(10, 10), (10, 9), (8, 8)])
        trades = {"A": [{"entry_ts": candles[0]["timestamp"], "exit_ts": candles[2]["timestamp"],
                         "entry_price": 10, "exit_price": 8, "hours": 48}]}
        profits = coin_short_profits(trades, {"A": candles}, "coinbase adv1 taker", 0)
        self.assertAlmostEqual(profits["A"][0], 18.65)

    def test_liquidated_trade_is_minus_100(self):
        candles = daily([(10, 10), (10, 25), (12, 12)])
        trades = {"A": [{"entry_ts": candles[0]["timestamp"], "exit_ts": candles[2]["timestamp"],
                         "entry_price": 10, "exit_price": 12, "hours": 48}]}
        self.assertEqual(coin_short_profits(trades, {"A": candles}, "kraken taker", 0)["A"], [-100])


class TestYearsAndPass(unittest.TestCase):
    def test_by_year(self):
        trades = {"A": [{"exit_time": "2022-03-01 00:00"}, {"exit_time": "2023-01-01 00:00"}],
                  "B": [{"exit_time": "2022-07-01 00:00"}]}
        profits = {"A": [10, -4], "B": [5]}
        self.assertEqual(by_year(trades, profits), {2022: 15, 2023: -4})

    def test_passes(self):
        self.assertTrue(passes([2] * 20 + [-1] * 10))

    def test_too_few(self):
        self.assertFalse(passes([2] * 19 + [-1] * 10))

    def test_all_zero_trades_fail(self):
        self.assertFalse(passes([0] * 30))

    def test_pf_exactly_1_1(self):
        self.assertFalse(passes([11] + [-1] * 10 + [0] * 19))


class TestYearRows(unittest.TestCase):
    def test_rows(self):
        longs = {2022: -10.0, 2024: 50.0}
        shorts_ = {2022: 30.0, 2023: -5.0}
        self.assertEqual(year_rows(longs, shorts_), [
            (2022, -10.0, 30.0, 20.0),
            (2023, 0.0, -5.0, -5.0),
            (2024, 50.0, 0.0, 50.0),
            ("total", 40.0, 25.0, 65.0),
        ])

    def test_empty(self):
        self.assertEqual(year_rows({}, {}), [("total", 0, 0, 0)])


class TestLongSide(unittest.TestCase):
    def test_long_side_uses_donchian_20_10(self):
        # rising 10 -> 20 over 300 days: the long rule buys once, at d21's open
        # (= close 20 = 10 + 200/299), still holding at the last close, 20.
        # Kraken taker: 99.62 / entry * 20 * 0.9962 - 100
        closes = [10 + i * 10 / 299 for i in range(300)]
        candles = daily([(10 if i == 0 else closes[i - 1], c) for i, c in enumerate(closes)])
        years = shorts.long_by_year({coin: candles for coin in COINS})
        entry = 10 + 20 * 10 / 299
        one = 100 * (1 - 0.0038) / entry * 20 * (1 - 0.0038) - 100
        self.assertEqual(list(years), [2023])
        self.assertAlmostEqual(years[2023], 6 * one)


class TestMain(unittest.TestCase):
    def write(self, folder, coins=COINS):
        # every coin falls steadily from 20 to 10 over 300 days
        closes = [20 - i * 10 / 299 for i in range(300)]
        candles = daily([(20 if i == 0 else closes[i - 1], c) for i, c in enumerate(closes)])
        for coin in coins:
            with open(os.path.join(folder, candle_file(coin, "1d")), "w") as f:
                f.write("timestamp,date,open,high,low,close,volume\n")
                for c in candles:
                    f.write(f"{c['timestamp']},{c['date']},{c['open']},{c['open']},"
                            f"{c['close']},{c['close']},1\n")

    def run_main(self, coins=COINS):
        with tempfile.TemporaryDirectory() as folder:
            self.write(folder, coins)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                shorts.main(folder)
        return out.getvalue()

    def test_missing_files(self):
        self.assertIn("missing 1d candle files", self.run_main(COINS[:2]))

    def test_steady_fall(self):
        out = self.run_main()
        # Each coin shorts once: the d20 close is the first below the previous 20,
        # so it shorts at d21's open = close 20 = 20 - 200/299, and is still short
        # at the last close, 10, after 278 days.
        entry = 20 - 20 * 10 / 299
        one = (100 * (entry - 10) / entry - 0.38 - 100 * 10 / entry * 0.0038
               - 100 * 0.0003 * 278)
        self.assertIn(f"CRV{1:>11}{one:>10.2f}{0:>12}", out)
        # 6 trades is under 30, one year, each coin's only trade is its best
        self.assertIn("FAIL  main test     6 trades", out)
        self.assertIn("FAIL  1 years", out)
        self.assertIn("FAIL  2 best trade", out)
        self.assertIn("pass  3 neighbors", out)
        self.assertIn("pass  4 high fees", out)
        self.assertIn("pass  5 late", out)
        self.assertIn("does not survive (3 of 6)", out)
        # the long side never buys a steady fall, so both = short
        self.assertIn(f"{'total':6}{0:>10.2f}{6 * one:>10.2f}{6 * one:>10.2f}", out)

    def test_neighbor_totals(self):
        # short 15/7: the d15 close is the first below the previous 15, so it
        # shorts at d16's open = close 15, held 283 days to the last close, 10
        out = self.run_main()
        entry = 20 - 15 * 10 / 299
        one = (100 * (entry - 10) / entry - 0.38 - 100 * 10 / entry * 0.0038
               - 100 * 0.0003 * 283)
        self.assertIn(f"15/7: {6 * one:.2f}", out)

    def test_both_column_adds_long_and_short(self):
        # pretend the long side made 10 in 2023: both = 10 + the shorts
        from unittest import mock
        with mock.patch.object(shorts, "long_by_year", return_value={2023: 10.0}):
            out = self.run_main()
        entry = 20 - 20 * 10 / 299
        one = (100 * (entry - 10) / entry - 0.38 - 100 * 10 / entry * 0.0038
               - 100 * 0.0003 * 278)
        self.assertIn(f"{2023:<6}{10:>10.2f}{6 * one:>10.2f}{10 + 6 * one:>10.2f}", out)
        self.assertIn(f"{'total':6}{10:>10.2f}{6 * one:>10.2f}{10 + 6 * one:>10.2f}", out)

    def test_no_funding_line(self):
        out = self.run_main()
        entry = 20 - 20 * 10 / 299
        one = 100 * (entry - 10) / entry - 0.38 - 100 * 10 / entry * 0.0038
        self.assertIn(f"no funding cost: {6 * one:.2f}", out)


if __name__ == "__main__":
    unittest.main()
