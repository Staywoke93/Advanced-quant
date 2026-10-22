import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import bot
from market_data import validate_ohlc
from backtest import backtest_frame


def sample_data(n=240, trend=0.001):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = 1.1 * (1 + trend) ** np.arange(n)
    return pd.DataFrame({
        "open": close * (1 - 0.0002),
        "high": close * (1 + 0.001),
        "low": close * (1 - 0.001),
        "close": close,
    }, index=idx)


class BotTests(unittest.TestCase):
    def test_ohlc_validation_accepts_valid_sample(self):
        result = validate_ohlc(sample_data())
        self.assertEqual(result["errors"], [])

    def test_ohlc_validation_rejects_impossible_candle(self):
        data = sample_data()
        data.iloc[-1, data.columns.get_loc("high")] = data.iloc[-1]["low"] * 0.5
        result = validate_ohlc(data)
        self.assertTrue(any("high below" in error for error in result["errors"]))

    def test_features_are_generated(self):
        f = bot.features(sample_data())
        self.assertGreater(len(f), 100)
        self.assertIn("atr_pct", f.columns)
        self.assertTrue(np.isfinite(f["atr_pct"].iloc[-1]))

    def test_session_inside_and_outside(self):
        inside = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)
        outside = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        self.assertTrue(bot.in_session(inside))
        self.assertFalse(bot.in_session(outside))

    def test_analyze_marks_outside_session(self):
        when = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        sig = bot.analyze("Test", "TEST", sample_data(), when)
        self.assertEqual(sig.status, "OUTSIDE_SESSION")

    def test_scan_can_analyse_before_configured_session(self):
        when = datetime(2025, 1, 1, 6, 30, tzinfo=timezone.utc)
        old_enforce = bot.ENFORCE_SESSION
        try:
            bot.ENFORCE_SESSION = False
            result = bot.scan(now=when, fetcher=lambda symbol: sample_data())
        finally:
            bot.ENFORCE_SESSION = old_enforce
        self.assertEqual(result["markets_requested"], len(bot.MARKETS))
        self.assertEqual(result["markets_analyzed"], len(bot.MARKETS))
        self.assertTrue(result["session_windows"])

    def test_upcoming_session_window_is_gmt_and_three_hours(self):
        now = datetime(2025, 1, 6, 6, 30, tzinfo=timezone.utc)
        windows = bot.upcoming_session_windows(now)
        self.assertTrue(windows)
        self.assertIn("London", windows[0]["event"])
        start = datetime.fromisoformat(windows[0]["suggested_entry_start_gmt"])
        end = datetime.fromisoformat(windows[0]["suggested_entry_end_gmt"])
        self.assertEqual((end - start).total_seconds(), 3 * 3600)

    def test_session_open_alert_detection(self):
        # Winter London/Frankfurt open is 08:00 GMT.
        now = datetime(2025, 1, 6, 8, 5, tzinfo=timezone.utc)
        events = bot.session_events(now)
        self.assertTrue(any(e["mode"] == "open" and e["name"] in ("London", "Frankfurt") for e in events))

    def test_one_hour_followup_detection(self):
        now = datetime(2025, 1, 6, 9, 5, tzinfo=timezone.utc)
        events = bot.session_events(now)
        self.assertTrue(any(e["mode"] == "scan" and e["name"] in ("London", "Frankfurt") for e in events))

    def test_three_market_overlap_detection(self):
        # At 13:05 GMT in winter, London and Frankfurt are open and New York has just opened.
        now = datetime(2025, 1, 6, 13, 5, tzinfo=timezone.utc)
        events = bot.session_events(now)
        self.assertTrue(any(e["mode"] == "overlap" for e in events))

    def test_invalid_config_rejected(self):
        old = bot.RISK_PER_TRADE
        try:
            bot.RISK_PER_TRADE = 0.5
            with self.assertRaises(ValueError):
                bot.validate_config()
        finally:
            bot.RISK_PER_TRADE = old

    def test_backtest_returns_metrics(self):
        result = backtest_frame(sample_data(), horizon=3, fee_bps=0)
        self.assertIn("trades", result)
        self.assertIn("max_drawdown_pct", result)
        self.assertGreaterEqual(result["trades"], 0)


if __name__ == "__main__":
    unittest.main()
