import contextlib
import io
import os
import tempfile
import unittest

import trend
from trend import (
    candle_file,
    close_open_trade,
    curve_drawdown,
    donchian_entry,
    donchian_exit,
    find_rule_trades,
    hold_curve,
    ma_up,
    make_rule,
    passes,
    strategy_curve,
)

H = 3_600_000


def make_candles(pairs):
    return [{"timestamp": h * H, "date": f"h{h}", "open": o, "close": c}
            for h, (o, c) in enumerate(pairs)]


class TestLockedSettings(unittest.TestCase):
    """Fixed before running anything. A failure here means someone changed them."""

    def test_rules(self):
        self.assertEqual(trend.RULES, [
            {"name": "donchian 20/10", "kind": "donchian", "entry": 20, "exit": 10},
            {"name": "donchian 55/20", "kind": "donchian", "entry": 55, "exit": 20},
            {"name": "ma 20/100", "kind": "ma", "fast": 20, "slow": 100},
            {"name": "ma 50/200", "kind": "ma", "fast": 50, "slow": 200},
        ])

    def test_coins_and_timeframes(self):
        self.assertEqual(trend.COINS, ["CRV", "BTC", "ETH", "UNI", "DOGE", "CAKE"])
        self.assertEqual(trend.TIMEFRAMES, ["1d", "4h"])

    def test_pass_rule(self):
        self.assertEqual(trend.PASS_FEES, "kraken taker")
        self.assertEqual(trend.PASS_MIN_TRADES, 30)
        self.assertEqual(trend.PASS_PROFIT_FACTOR, 1.1)


class TestDonchian(unittest.TestCase):
    def test_new_high_buys(self):
        # previous 3 closes 1, 2, 3: highest 3, close 4 is above
        self.assertTrue(donchian_entry([1, 2, 3, 4], 3, 3))

    def test_equal_to_high_does_not_buy(self):
        self.assertFalse(donchian_entry([1, 3, 2, 3], 3, 3))

    def test_oldest_of_the_n_closes_counts(self):
        # previous 3 closes 5, 1, 2: highest is the oldest, 5. Close 3 isn't above it
        self.assertFalse(donchian_entry([5, 1, 2, 3], 3, 3))

    def test_needs_n_previous_closes(self):
        self.assertFalse(donchian_entry([1, 2, 30], 2, 3))

    def test_new_low_sells(self):
        # previous 3 closes 5, 4, 6: lowest 4, close 3 is below
        self.assertTrue(donchian_exit([5, 4, 6, 3], 3, 3))

    def test_equal_to_low_does_not_sell(self):
        self.assertFalse(donchian_exit([5, 4, 6, 4], 3, 3))

    def test_oldest_of_the_m_closes_counts(self):
        # previous 3 closes 1, 5, 6: lowest is the oldest, 1. Close 3 isn't below it
        self.assertFalse(donchian_exit([1, 5, 6, 3], 3, 3))

    def test_exit_needs_m_previous_closes(self):
        self.assertFalse(donchian_exit([5, 1], 1, 3))


class TestMovingAverages(unittest.TestCase):
    def test_up(self):
        # fast (4 + 5) / 2 = 4.5, slow (2 + 3 + 4 + 5) / 4 = 3.5
        self.assertTrue(ma_up([1, 2, 3, 4, 5], 4, 2, 4))

    def test_down(self):
        self.assertFalse(ma_up([5, 4, 3, 2, 1], 4, 2, 4))

    def test_equal_is_not_up(self):
        self.assertFalse(ma_up([3, 3, 3, 3], 3, 2, 4))

    def test_unknown_until_slow_window_fills(self):
        self.assertIsNone(ma_up([1, 2, 3], 2, 2, 4))


class TestMakeRule(unittest.TestCase):
    def test_donchian(self):
        buy, sell = make_rule({"kind": "donchian", "entry": 3, "exit": 3})
        self.assertTrue(buy([1, 2, 3, 4], 3))
        self.assertTrue(sell([5, 4, 6, 3], 3))

    def test_ma(self):
        buy, sell = make_rule({"kind": "ma", "fast": 2, "slow": 4})
        self.assertTrue(buy([1, 2, 3, 4, 5], 4))
        self.assertFalse(sell([1, 2, 3, 4, 5], 4))
        self.assertTrue(sell([5, 4, 3, 2, 1], 4))

    def test_ma_unknown_trend_neither_buys_nor_sells(self):
        buy, sell = make_rule({"kind": "ma", "fast": 2, "slow": 4})
        self.assertFalse(buy([1, 2, 3], 2))
        self.assertFalse(sell([1, 2, 3], 2))

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            make_rule({"kind": "rsi"})


class TestFindRuleTrades(unittest.TestCase):
    # donchian 3/2:
    #   i3 close 13 > highest of [10, 11, 12] -> buy at i4 open 13.5
    #   i4 close 14, lowest of [12, 13] is 12 -> hold
    #   i5 close 12 < lowest of [13, 14] -> sell at i6 open 11.5
    DONCHIAN = [(10, 10), (10, 11), (11, 12), (12, 13), (13.5, 14), (14, 12), (11.5, 11.5)]

    def test_donchian_trade(self):
        buy, sell = make_rule({"kind": "donchian", "entry": 3, "exit": 2})
        trades, open_trade = find_rule_trades(make_candles(self.DONCHIAN), buy, sell)
        self.assertEqual(trades, [{
            "entry_ts": 4 * H, "exit_ts": 6 * H,
            "entry_time": "h4", "entry_price": 13.5,
            "exit_time": "h6", "exit_price": 11.5,
            "reason": "signal", "hours": 2,
        }])
        self.assertIsNone(open_trade)

    def test_ma_trade(self):
        # ma 2/3: i2 fast (10 + 13) / 2 = 11.5 > slow 11 -> buy at i3 open 12.5
        # i3: fast 12.5 > slow 11.67 -> hold
        # i4: fast (12 + 9) / 2 = 10.5 < slow (13 + 12 + 9) / 3 = 11.33 -> sell at i5 open 9.5
        candles = make_candles([(10, 10), (10, 10), (10, 13), (12.5, 12), (12, 9), (9.5, 9.5)])
        buy, sell = make_rule({"kind": "ma", "fast": 2, "slow": 3})
        trades, _ = find_rule_trades(candles, buy, sell)
        self.assertEqual([(t["entry_price"], t["exit_price"], t["exit_time"]) for t in trades],
                         [(12.5, 9.5, "h5")])

    def test_still_holding_at_the_end(self):
        # same entry, then i5 close 15: lowest of [13, 14] is 13 -> hold
        candles = make_candles(self.DONCHIAN[:5] + [(14, 15)])
        buy, sell = make_rule({"kind": "donchian", "entry": 3, "exit": 2})
        trades, open_trade = find_rule_trades(candles, buy, sell)
        self.assertEqual(trades, [])
        self.assertEqual(open_trade["entry_price"], 13.5)

    def test_open_trade_counted_at_last_close(self):
        candles = make_candles(self.DONCHIAN[:5] + [(14, 15)])
        buy, sell = make_rule({"kind": "donchian", "entry": 3, "exit": 2})
        trades, open_trade = find_rule_trades(candles, buy, sell)
        self.assertEqual(close_open_trade(trades, open_trade, candles), [{
            "entry_ts": 4 * H, "exit_ts": 5 * H,
            "entry_time": "h4", "entry_price": 13.5,
            "exit_time": "h5", "exit_price": 15,
            "reason": "still open", "hours": 1,
        }])

    def test_nothing_open(self):
        self.assertEqual(close_open_trade(["t"], None, make_candles([(1, 1)])), ["t"])


class TestCurves(unittest.TestCase):
    CANDLES = make_candles([(10, 10), (10, 12), (12, 9), (9, 11)])

    def test_strategy_curve_values_the_open_trade(self):
        # bought at h1 open 10, sold at h3 open 9, no fees ($100 = 10 coins)
        # h0: not in yet = 0. h1 close 12: +20. h2 close 9: -10. h3: sold at 9: -10
        trades = [{"entry_ts": 1 * H, "exit_ts": 3 * H, "entry_price": 10, "exit_price": 9}]
        curve = strategy_curve(self.CANDLES, trades, 0, 0)
        self.assertEqual([round(v, 6) for v in curve], [0, 20, -10, -10])
        # from the +20 high down to -10: a 30 drawdown, not just the 10 lost
        self.assertEqual(round(curve_drawdown(curve), 6), 30)

    def test_strategy_curve_two_trades(self):
        # 10 -> 12 (h0 to h2, +20), then 9 -> 11 (h2 to h3 last close, +22.22)
        trades = [{"entry_ts": 0, "exit_ts": 2 * H, "entry_price": 10, "exit_price": 12},
                  {"entry_ts": 2 * H, "exit_ts": 3 * H, "entry_price": 9, "exit_price": 11}]
        curve = strategy_curve(self.CANDLES, trades, 0, 0)
        # h0 close 10: 0. h1 close 12: +20. h2: first closed +20, second in at 9,
        # close 9: +0 -> 20. h3: second closed at 11: 20 + 22.222 = 42.222
        self.assertEqual([round(v, 3) for v in curve], [0, 20, 20, 42.222])

    def test_strategy_curve_is_zero_before_buying(self):
        # h0 close 15 happens before the buy at h1 open 10, so it's worth 0 there
        candles = make_candles([(10, 15), (10, 12), (12, 12)])
        trades = [{"entry_ts": 1 * H, "exit_ts": 2 * H, "entry_price": 10, "exit_price": 12}]
        curve = strategy_curve(candles, trades, 0, 0)
        self.assertEqual([round(v, 6) for v in curve], [0, 20, 20])

    def test_hold_curve(self):
        # $100 at 10 = 10 coins; closes 12, 8, 15
        candles = make_candles([(10, 12), (12, 8), (8, 15)])
        self.assertEqual([round(v, 6) for v in hold_curve(candles, 0, 0)], [20, -20, 50])
        self.assertEqual(round(curve_drawdown(hold_curve(candles, 0, 0)), 6), 40)

    def test_drawdown_starts_from_zero(self):
        self.assertEqual(curve_drawdown([-5, -3]), 5)

    def test_drawdown_empty(self):
        self.assertEqual(curve_drawdown([]), 0)


class TestPasses(unittest.TestCase):
    def test_passes(self):
        # 30 trades, wins 40, losses 10: PF 4
        self.assertTrue(passes([2] * 20 + [-1] * 10))

    def test_29_trades_is_too_few(self):
        self.assertFalse(passes([2] * 19 + [-1] * 10))

    def test_profit_factor_exactly_1_1_fails(self):
        # wins 11, losses 10
        self.assertFalse(passes([11] + [-1] * 10 + [0] * 19))

    def test_all_zero_trades_fail(self):
        self.assertFalse(passes([0] * 30))


class TestRun(unittest.TestCase):
    def test_candle_file(self):
        self.assertEqual(candle_file("CRV", "4h"), "candles_coinbase_crv_usd_4h.csv")

    def run_folder(self, write_coins):
        """Price rises steadily from 10 to 20 over 300 candles for each coin."""
        closes = [10 + i * 10 / 299 for i in range(300)]
        closes[-1] = 20.0
        with tempfile.TemporaryDirectory() as folder:
            for coin in write_coins:
                with open(os.path.join(folder, candle_file(coin, "1d")), "w") as f:
                    f.write("timestamp,date,open,high,low,close,volume\n")
                    for i, c in enumerate(closes):
                        o = 10 if i == 0 else closes[i - 1]
                        f.write(f"{i * 24 * H},d{i},{o},{c},{o},{c},1\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ok = trend.run_timeframe("1d", folder)
        return ok, out.getvalue()

    def test_missing_files(self):
        ok, out = self.run_folder(["CRV", "BTC"])
        self.assertFalse(ok)
        self.assertIn("missing candles_coinbase_eth_usd_1d.csv", out)

    def test_full_run(self):
        ok, out = self.run_folder(trend.COINS)
        self.assertTrue(ok)
        rows = [line.split() for line in out.splitlines()]
        # a steady rise: each rule buys once and is still holding at the end,
        # so 1 trade per coin = 6 trades, fewer than 30 -> fail
        results = [r for r in rows if r and r[-1] in ("PASS", "fail")]
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r[8], "6")
            self.assertEqual(r[-1], "fail")
        # hold: 99.62 / 10 = 9.962 coins; * 20 = 199.24; * 0.9962 = 198.4829 -> 98.48
        hold_rows = [r for r in rows if r[:3] == ["buy", "and", "hold"]]
        self.assertEqual(hold_rows[0][3:], ["98.48"] * 6)
        # hold drawdown: worth 99.62 / 10 * 10 * 0.9962 = 99.2414 at the first close,
        # -0.76 from the starting 0, and never lower after that
        self.assertEqual(hold_rows[1][3:], ["0.76"] * 6)


if __name__ == "__main__":
    unittest.main()
