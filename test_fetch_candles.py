import contextlib
import csv
import io
import os
import tempfile
import unittest
from unittest import mock

import fetch_candles
from fetch_candles import (
    combine_candles,
    csv_name,
    date_to_ms,
    drop_unfinished,
    fetch_all,
    fetch_timeframe,
    merge_candles,
    ms_to_date,
    next_since,
    write_csv,
)

H = 3_600_000  # one hour in milliseconds


def candle(ts, close=1.0):
    """A candle at time ts. Only the timestamp matters for most tests."""
    return [ts, close, close, close, close, 100.0]


class FakeCoinbase:
    """Answers like Coinbase: candles inside the window
    since .. since + limit hours (end excluded)."""

    def __init__(self, candles):
        self.candles = candles
        self.calls = 0

    def parse_timeframe(self, timeframe):
        return 3600  # seconds in 1h

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        if self.calls > 50:
            raise RuntimeError("fetch_all is stuck in a loop")
        end = since + limit * H
        return [c for c in self.candles if since <= c[0] < end]


class FakeCoinbaseInclusiveEnd(FakeCoinbase):
    """Same, but the end of the window is included (edge candle can repeat)."""

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        end = since + limit * H
        return [c for c in self.candles if since <= c[0] <= end]


class FakeNewestFirst(FakeCoinbase):
    """Same window as Coinbase, but sends candles newest-first."""

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        batch = super().fetch_ohlcv(symbol, timeframe, since, limit)
        return list(reversed(batch))


class FakeKraken(FakeCoinbase):
    """Ignores 'since' and always returns its newest 4 candles."""

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        if self.calls > 50:
            raise RuntimeError("fetch_all is stuck in a loop")
        return self.candles[-4:]


def times(candles):
    """Candle timestamps in hours, e.g. [0, 1, 2]. Easier to read in asserts."""
    return [c[0] // H for c in candles]


class TestDates(unittest.TestCase):
    def test_one_day_after_1970(self):
        self.assertEqual(date_to_ms("1970-01-02"), 86_400_000)

    def test_real_date(self):
        # 2023-10-01 00:00 UTC = 1,696,118,400 seconds
        self.assertEqual(date_to_ms("2023-10-01"), 1_696_118_400_000)

    def test_ms_to_date_midnight(self):
        self.assertEqual(ms_to_date(1_696_118_400_000), "2023-10-01 00:00")

    def test_ms_to_date_afternoon(self):
        self.assertEqual(ms_to_date(1_696_118_400_000 + 13 * H), "2023-10-01 13:00")


class TestNextSince(unittest.TestCase):
    def test_after_full_batch_starts_one_candle_past_newest(self):
        batch = [candle(8 * H), candle(9 * H), candle(10 * H)]
        self.assertEqual(next_since(batch, 8 * H, H, 3), 11 * H)

    def test_newest_first_batch_uses_newest_candle(self):
        batch = [candle(10 * H), candle(9 * H), candle(8 * H)]
        self.assertEqual(next_since(batch, 8 * H, H, 3), 11 * H)

    def test_after_empty_batch_skips_one_window(self):
        # since 5h, 3 candles per window -> next window starts at 8h
        self.assertEqual(next_since([], 5 * H, H, 3), 8 * H)


class TestMerge(unittest.TestCase):
    def test_sorts_by_time(self):
        merged = merge_candles([candle(3 * H), candle(1 * H), candle(2 * H)])
        self.assertEqual(times(merged), [1, 2, 3])

    def test_removes_duplicates_and_keeps_later_copy(self):
        merged = merge_candles([candle(1 * H, close=5.0), candle(1 * H, close=6.0)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][4], 6.0)

    def test_empty(self):
        self.assertEqual(merge_candles([]), [])


class TestDropUnfinished(unittest.TestCase):
    def test_drops_candle_still_forming(self):
        # it's 2:30; the 2:00 candle isn't done until 3:00
        candles = [candle(0), candle(1 * H), candle(2 * H)]
        kept = drop_unfinished(candles, H, 2 * H + H // 2)
        self.assertEqual(times(kept), [0, 1])

    def test_keeps_candle_that_just_finished(self):
        # it's exactly 3:00, so the 2:00 candle is final
        candles = [candle(0), candle(1 * H), candle(2 * H)]
        kept = drop_unfinished(candles, H, 3 * H)
        self.assertEqual(times(kept), [0, 1, 2])


class TestFetchAll(unittest.TestCase):
    def test_pages_through_all_candles(self):
        # candles 0h..9h, 3 per request: windows 0-3, 3-6, 6-9, 9-12
        ex = FakeCoinbase([candle(h * H) for h in range(10)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 10 * H, limit=3)
        self.assertEqual(times(result), list(range(10)))
        self.assertEqual(ex.calls, 4)

    def test_keeps_going_past_a_gap(self):
        # outage from 2h to 6h: windows 0-3 [0,1], 2-5 [], 5-8 [7], 8-11 [8]
        ex = FakeCoinbase([candle(0), candle(1 * H), candle(7 * H), candle(8 * H)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 9 * H, limit=3)
        self.assertEqual(times(result), [0, 1, 7, 8])

    def test_start_before_coin_was_listed(self):
        # listed at 6h; windows 0-3 [], 3-6 [], 6-9 [6, 7]
        ex = FakeCoinbase([candle(6 * H), candle(7 * H)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 8 * H, limit=3)
        self.assertEqual(times(result), [6, 7])

    def test_no_duplicates_when_window_end_is_included(self):
        ex = FakeCoinbaseInclusiveEnd([candle(h * H) for h in range(10)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 10 * H, limit=3)
        self.assertEqual(times(result), list(range(10)))

    def test_newest_first_exchange_ends_up_oldest_first(self):
        # windows 0-3 [2,1,0], 3-6 [5,4,3], 6-9 [8,7,6], 9-12 [9]
        ex = FakeNewestFirst([candle(h * H) for h in range(10)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 10 * H, limit=3)
        self.assertEqual(times(result), list(range(10)))
        self.assertEqual(ex.calls, 4)

    def test_unfinished_last_candle_is_dropped(self):
        # it's 9:30, so the 9:00 candle is still forming
        ex = FakeCoinbase([candle(h * H) for h in range(10)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 9 * H + H // 2, limit=3)
        self.assertEqual(times(result), list(range(9)))

    def test_exchange_that_ignores_since(self):
        # Kraken-style: always the newest 4 (6h..9h), then stops
        ex = FakeKraken([candle(h * H) for h in range(10)])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 10 * H, limit=3)
        self.assertEqual(times(result), [6, 7, 8, 9])
        self.assertEqual(ex.calls, 1)

    def test_candles_older_than_since_are_ignored(self):
        # asks from 8h; Kraken sends 6h..9h; only 8h and 9h are wanted
        ex = FakeKraken([candle(h * H) for h in range(10)])
        result = fetch_all(ex, "CRV/USD", "1h", 8 * H, 10 * H, limit=3)
        self.assertEqual(times(result), [8, 9])

    def test_nothing_to_fetch(self):
        ex = FakeCoinbase([])
        result = fetch_all(ex, "CRV/USD", "1h", 0, 6 * H, limit=3)
        self.assertEqual(result, [])
        self.assertEqual(ex.calls, 2)  # windows 0-3 and 3-6, both empty


class TestCsv(unittest.TestCase):
    def test_writes_header_and_rows(self):
        candles = [
            [1_696_118_400_000, 0.45, 0.47, 0.44, 0.46, 1200.5],
            [1_696_118_400_000 + H, 0.46, 0.48, 0.45, 0.47, 980.0],
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.csv")
            write_csv(candles, path)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["timestamp", "date", "open", "high", "low", "close", "volume"])
        self.assertEqual(rows[1], ["1696118400000", "2023-10-01 00:00", "0.45", "0.47", "0.44", "0.46", "1200.5"])
        self.assertEqual(rows[2][1], "2023-10-01 01:00")
        self.assertEqual(len(rows), 3)

    def test_csv_name(self):
        self.assertEqual(csv_name("coinbase", "CRV/USD", "1h"), "candles_coinbase_crv_usd_1h.csv")

    def test_csv_name_start_only_keeps_plain_name(self):
        self.assertEqual(csv_name("coinbase", "CRV/USD", "1h", "2023-10-01"),
                         "candles_coinbase_crv_usd_1h.csv")

    def test_csv_name_with_end_date(self):
        self.assertEqual(csv_name("coinbase", "CRV/USD", "1h", "2021-10-01", "2023-10-01"),
                         "candles_coinbase_crv_usd_1h_2021-10-01_to_2023-10-01.csv")


def hourly(h, o, hi, lo, c, v):
    return [h * H, o, hi, lo, c, v]


class TestCombine(unittest.TestCase):
    FOUR = 4 * H

    def test_two_blocks(self):
        candles = [
            hourly(0, 1.0, 1.5, 0.9, 1.2, 10), hourly(1, 1.2, 1.8, 1.1, 1.7, 20),
            hourly(2, 1.7, 1.9, 1.0, 1.1, 30), hourly(3, 1.1, 1.3, 0.8, 1.25, 40),
            hourly(4, 1.25, 1.4, 1.2, 1.3, 5), hourly(5, 1.3, 2.0, 1.3, 1.9, 5),
            hourly(6, 1.9, 1.95, 1.6, 1.6, 5), hourly(7, 1.6, 1.7, 1.5, 1.55, 5),
        ]
        self.assertEqual(combine_candles(candles, self.FOUR), [
            # open of hour 0, top high 1.9, bottom low 0.8, close of hour 3, 10+20+30+40
            [0, 1.0, 1.9, 0.8, 1.25, 100],
            [4 * H, 1.25, 2.0, 1.2, 1.55, 20],
        ])

    def test_missing_hour_inside_a_block(self):
        # hour 2 missing: the block is made from hours 0, 1, 3
        candles = [hourly(0, 1, 2, 1, 1.5, 1), hourly(1, 1.5, 3, 1.4, 2, 1),
                   hourly(3, 2, 2.5, 0.5, 1, 1)]
        self.assertEqual(combine_candles(candles, self.FOUR), [[0, 1, 3, 0.5, 1, 3]])

    def test_start_not_on_a_block_edge(self):
        # hours 2, 3 belong to the 00:00 block; 4, 5 to the 04:00 block
        candles = [hourly(h, h, h, h, h, 1) for h in (2, 3, 4, 5)]
        self.assertEqual(combine_candles(candles, self.FOUR),
                         [[0, 2, 3, 2, 3, 2], [4 * H, 4, 5, 4, 5, 2]])

    def test_empty(self):
        self.assertEqual(combine_candles([], self.FOUR), [])

    def test_does_not_change_the_input(self):
        candles = [hourly(0, 1, 2, 1, 1.5, 1), hourly(1, 1.5, 3, 1.4, 2, 1)]
        combine_candles(candles, self.FOUR)
        self.assertEqual(candles[0], [0, 1, 2, 1, 1.5, 1])


class FakeWithTimeframes(FakeCoinbase):
    """Coinbase-like windows, with a list of offered candle sizes."""

    def __init__(self, candles, offered):
        super().__init__(candles)
        self.timeframes = {tf: tf for tf in offered}
        self.asked = []

    def parse_timeframe(self, timeframe):
        return {"1h": 3600, "4h": 14400, "1d": 86400}[timeframe]

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.asked.append(timeframe)
        return super().fetch_ohlcv(symbol, timeframe, since, limit)


class TestFetchTimeframe(unittest.TestCase):
    def test_builds_4h_from_1h_when_not_offered(self):
        # hours 0..9, now is 10:00: blocks 00:00 (0-3), 04:00 (4-7), 08:00 (8-9)
        # the 08:00 block isn't done until 12:00, so it's dropped
        hours = [[h * H, 1, h + 1, 0.5, h, 1] for h in range(10)]
        ex = FakeWithTimeframes(hours, ["1h", "1d"])
        result = fetch_timeframe(ex, "CRV/USD", "4h", 0, 10 * H)
        self.assertEqual(result, [[0, 1, 4, 0.5, 3, 4], [4 * H, 1, 8, 0.5, 7, 4]])
        self.assertEqual(set(ex.asked), {"1h"})

    def test_asks_directly_when_offered(self):
        ex = FakeWithTimeframes([candle(0)], ["1h", "4h"])
        with contextlib.redirect_stdout(io.StringIO()):
            fetch_timeframe(ex, "CRV/USD", "4h", 0, 300 * 4 * H)
        self.assertEqual(set(ex.asked), {"4h"})


class TestMain(unittest.TestCase):
    """main() with the exchange and fetch_all faked, so no network is used."""

    def run_main(self, args):
        calls = []

        def fake_fetch_timeframe(exchange, symbol, timeframe, since_ms, now_ms):
            calls.append((exchange, symbol, timeframe, since_ms))
            self.end_ms = now_ms
            return [candle(since_ms), candle(since_ms + H)]

        fake_ccxt = mock.Mock()
        fake_ccxt.coinbase.return_value = "coinbase exchange"
        fake_ccxt.kraken.return_value = "kraken exchange"
        with tempfile.TemporaryDirectory() as folder:
            old = os.getcwd()
            os.chdir(folder)
            try:
                with mock.patch.object(fetch_candles, "ccxt", fake_ccxt), \
                     mock.patch.object(fetch_candles, "fetch_timeframe", fake_fetch_timeframe), \
                     contextlib.redirect_stdout(io.StringIO()):
                    fetch_candles.main(args)
                files = sorted(os.listdir("."))
            finally:
                os.chdir(old)
        return calls, files

    def test_defaults(self):
        calls, files = self.run_main([])
        self.assertEqual(calls, [("coinbase exchange", "CRV/USD", "1h", date_to_ms("2023-10-01"))])
        self.assertEqual(files, ["candles_coinbase_crv_usd_1h.csv"])

    def test_start_date_and_exchange(self):
        calls, files = self.run_main(["2026-09-29", "kraken"])
        self.assertEqual(calls, [("kraken exchange", "CRV/USD", "1h", date_to_ms("2026-09-29"))])
        self.assertEqual(files, ["candles_kraken_crv_usd_1h.csv"])

    def test_symbol(self):
        calls, files = self.run_main(["2023-10-01", "coinbase", "BTC/USD"])
        self.assertEqual(calls[0][1], "BTC/USD")
        self.assertEqual(files, ["candles_coinbase_btc_usd_1h.csv"])

    def test_end_date_stops_fetch_and_names_file(self):
        calls, files = self.run_main(["2021-10-01", "coinbase", "CRV/USD", "2023-10-01"])
        self.assertEqual(calls[0][3], date_to_ms("2021-10-01"))
        self.assertEqual(self.end_ms, date_to_ms("2023-10-01"))
        self.assertEqual(files, ["candles_coinbase_crv_usd_1h_2021-10-01_to_2023-10-01.csv"])

    def test_now_means_no_end_date_and_timeframe_arg(self):
        with mock.patch.object(fetch_candles.time, "time", return_value=1_800_000_000):
            calls, files = self.run_main(["2021-10-01", "coinbase", "BTC/USD", "now", "1d"])
        self.assertEqual(calls[0][1:3], ("BTC/USD", "1d"))
        self.assertEqual(self.end_ms, 1_800_000_000_000)
        self.assertEqual(files, ["candles_coinbase_btc_usd_1d.csv"])

    def test_timeframe_with_end_date(self):
        calls, files = self.run_main(["2021-10-01", "coinbase", "CRV/USD", "2023-10-01", "4h"])
        self.assertEqual(calls[0][2], "4h")
        self.assertEqual(files, ["candles_coinbase_crv_usd_4h_2021-10-01_to_2023-10-01.csv"])

    def test_no_end_date_fetches_to_now(self):
        with mock.patch.object(fetch_candles.time, "time", return_value=1_800_000_000):
            self.run_main([])
        self.assertEqual(self.end_ms, 1_800_000_000_000)

    def test_end_date_in_future_stops_at_now(self):
        # 1,800,000,000 s is Jan 2027; an end date in 2030 can't fetch past now
        with mock.patch.object(fetch_candles.time, "time", return_value=1_800_000_000):
            self.run_main(["2026-01-01", "coinbase", "CRV/USD", "2030-01-01"])
        self.assertEqual(self.end_ms, 1_800_000_000_000)


if __name__ == "__main__":
    unittest.main()
