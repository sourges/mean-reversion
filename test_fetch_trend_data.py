import unittest
from unittest import mock

import fetch_trend_data


class TestJobs(unittest.TestCase):
    def test_twelve_files(self):
        self.assertEqual(len(fetch_trend_data.jobs()), 12)

    def test_first_and_last(self):
        jobs = fetch_trend_data.jobs()
        self.assertEqual(jobs[0], ["2021-10-01", "coinbase", "CRV/USD", "now", "1d"])
        self.assertEqual(jobs[1], ["2021-10-01", "coinbase", "CRV/USD", "now", "4h"])
        self.assertEqual(jobs[-1], ["2021-10-01", "coinbase", "CAKE/USD", "now", "4h"])

    def test_same_coins_and_sizes_as_the_trend_test(self):
        import trend
        self.assertEqual(fetch_trend_data.COINS, trend.COINS)
        self.assertEqual(fetch_trend_data.TIMEFRAMES, trend.TIMEFRAMES)

    def test_main_fetches_every_job(self):
        with mock.patch.object(fetch_trend_data.fetch_candles, "main") as fake_main, \
             mock.patch("builtins.print"):
            fetch_trend_data.main()
        self.assertEqual([c.args[0] for c in fake_main.call_args_list], fetch_trend_data.jobs())


if __name__ == "__main__":
    unittest.main()
