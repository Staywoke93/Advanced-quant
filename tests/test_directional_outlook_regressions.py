"""Regression tests for directional labels, review windows, and signal caps."""
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import bot


def sample_data(n=240, trend=0.001):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = 1.1 * (1 + trend) ** np.arange(n)
    return pd.DataFrame({
        "open": close * (1 - 0.0002),
        "high": close * (1 + 0.001),
        "low": close * (1 - 0.001),
        "close": close,
    }, index=idx)


class DirectionalOutlookRegressionTests(unittest.TestCase):
    def _line(self, score, direction="LONG", reasons=None):
        return bot.format_directional_outlook({
            "market": "Test", "direction": direction,
            "forecast_outlook": "RISE" if direction == "LONG" else "FALL",
            "status": "NO_TRADE", "last_price": 1.2345,
            "confidence_score": score,
            "reasons": reasons or ["EMA trend alignment bullish"],
        })

    def test_score_threshold_boundaries(self):
        self.assertIn("STRONG BUY", self._line(0.80))
        self.assertIn("BUY", self._line(0.79))
        self.assertIn("BUY", self._line(0.65))
        self.assertIn("WEAK BUY", self._line(0.649))

    def test_sell_threshold_boundaries(self):
        self.assertIn("STRONG SELL", self._line(0.80, "SHORT", ["EMA trend alignment bearish"]))
        self.assertIn("SELL", self._line(0.65, "SHORT", ["EMA trend alignment bearish"]))
        self.assertIn("WEAK SELL", self._line(0.649, "SHORT", ["EMA trend alignment bearish"]))

    def test_conflict_overrides_high_score(self):
        line = self._line(0.95, "SHORT", [
            "EMA trend alignment bearish", "3-bar momentum positive"
        ])
        self.assertIn("WAIT", line)
        self.assertIn("TRADE APPROVAL: NOT ACTIONABLE (NO_TRADE)", line)

    def test_directional_label_does_not_imply_trade_approval(self):
        line = self._line(0.90)
        self.assertIn("STRONG BUY", line)
        self.assertIn("TRADE APPROVAL: NOT ACTIONABLE (NO_TRADE)", line)
        self.assertIn("current price 1.2345", line)
        self.assertIn("reasons:", line)

    def test_signal_cap_does_not_remove_market_outlooks(self):
        when = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)
        old_cap, old_session = bot.MAX_SIMULTANEOUS_SIGNALS, bot.ENFORCE_SESSION
        try:
            bot.MAX_SIMULTANEOUS_SIGNALS = 1
            bot.ENFORCE_SESSION = False
            result = bot.scan(now=when, fetcher=lambda symbol: sample_data())
        finally:
            bot.MAX_SIMULTANEOUS_SIGNALS, bot.ENFORCE_SESSION = old_cap, old_session

        self.assertEqual(result["markets_requested"], len(bot.MARKETS))
        self.assertEqual(result["markets_analyzed"], len(bot.MARKETS))
        self.assertEqual(len(result["signals"]), len(bot.MARKETS))
        self.assertTrue(any(
            "concurrent-signal cap reached" in " ".join(s.get("reasons", []))
            for s in result["signals"]
        ))
        # Every analysed market should still be formattable for the outlook section.
        for signal in result["signals"]:
            self.assertTrue(bot.format_directional_outlook(signal))


if __name__ == "__main__":
    unittest.main()
