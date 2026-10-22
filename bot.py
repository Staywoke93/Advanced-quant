#!/usr/bin/env python3
"""Conservative three-hour market research scanner. Signals only; no orders are placed."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("advanced-quant-bot")

MARKETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "Gold futures proxy": "GC=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
}

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "1000"))
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "USD")
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.0025"))  # conservative 0.25%
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.01"))  # 1% of initial equity
MAX_SIMULTANEOUS_SIGNALS = int(os.getenv("MAX_SIMULTANEOUS_SIGNALS", "2"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.64"))
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0003"))
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "0.025"))
HORIZON_BARS = int(os.getenv("HORIZON_BARS", "3"))
SESSION_START_UTC = os.getenv("SESSION_START_UTC", "12:00")
SESSION_END_UTC = os.getenv("SESSION_END_UTC", "15:00")
ENFORCE_SESSION = os.getenv("ENFORCE_SESSION", "true").lower() == "true"
DATA_INTERVAL = os.getenv("DATA_INTERVAL", "1h")
DATA_PERIOD = os.getenv("DATA_PERIOD", "730d")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "").strip()
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOKEN = os.getenv("NTFY_TOKEN", "").strip()
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))


@dataclass
class Signal:
    market: str
    symbol: str
    timestamp_utc: str
    last_price: float
    direction: str
    confidence_score: float
    confidence_is_probability: bool
    horizon_hours: int
    session_utc: str
    atr_pct: float
    stop_distance_pct: float
    target_distance_pct: float
    risk_budget: float
    suggested_notional: float
    status: str
    reasons: list[str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def fetch_history(symbol: str, period: str = DATA_PERIOD, interval: str = DATA_INTERVAL,
                  retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch OHLC data with retries, basic schema validation and stale/empty-data rejection."""
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True,
                             progress=False, threads=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty provider response")
            df = _flatten_columns(df)
            required = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in required):
                raise ValueError(f"missing OHLC columns: {list(df.columns)}")
            df = df[required].copy()
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) < 120:
                raise ValueError(f"not enough bars ({len(df)})")
            if (df[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("non-positive OHLC values")
            return df
        except Exception as exc:
            wait = 2 ** attempt + random.random()
            log.warning("%s fetch attempt %d/%d failed: %s", symbol, attempt + 1, retries, exc)
            if attempt + 1 < retries:
                time.sleep(wait)
    return None


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # Handle zero-loss and flat windows explicitly; otherwise a monotonic
    # rising series produces NaN RSI values and drops every feature row.
    rs = up / down.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((down == 0) & (up > 0), 100.0)
    result = result.mask((down == 0) & (up == 0), 50.0)
    return result


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features only: each row uses that row and earlier observations."""
    out = df.copy()
    c = out["close"]
    out["ret_1"] = c.pct_change()
    out["ret_3"] = c.pct_change(3)
    out["ret_6"] = c.pct_change(6)
    for n in (9, 21, 50):
        out[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()
    out["rsi"] = rsi(c)
    prev = c.shift(1)
    tr = pd.concat([(out["high"] - out["low"]), (out["high"] - prev).abs(),
                    (out["low"] - prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr"] / c
    out["vol_24"] = out["ret_1"].rolling(24).std()
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def score_row(row: pd.Series) -> tuple[str, float, list[str]]:
    """Rule-based score. Returned confidence_score is a ranking heuristic, not a probability."""
    score = 0.0
    reasons: list[str] = []
    if row["ema_9"] > row["ema_21"] > row["ema_50"]:
        score += 0.34
        reasons.append("EMA trend alignment bullish")
    elif row["ema_9"] < row["ema_21"] < row["ema_50"]:
        score -= 0.34
        reasons.append("EMA trend alignment bearish")
    if row["ret_3"] > 0:
        score += min(0.18, abs(float(row["ret_3"])) * 18)
        reasons.append("3-bar momentum positive")
    elif row["ret_3"] < 0:
        score -= min(0.18, abs(float(row["ret_3"])) * 18)
        reasons.append("3-bar momentum negative")
    if row["ret_6"] > 0:
        score += min(0.12, abs(float(row["ret_6"])) * 10)
    elif row["ret_6"] < 0:
        score -= min(0.12, abs(float(row["ret_6"])) * 10)
    rv = float(row["rsi"])
    if 50 <= rv <= 68:
        score += 0.12
        reasons.append(f"RSI supportive ({rv:.1f})")
    elif 32 <= rv < 50:
        score -= 0.12
        reasons.append(f"RSI supports bearish momentum ({rv:.1f})")
    elif rv >= 75:
        score -= 0.06
        reasons.append(f"overbought caution ({rv:.1f})")
    elif rv <= 25:
        score += 0.06
        reasons.append(f"oversold caution ({rv:.1f})")
    direction = "LONG" if score > 0 else "SHORT" if score < 0 else "NO_TRADE"
    confidence = min(0.90, 0.50 + abs(score) * 0.70)
    return direction, confidence, reasons


def in_session(now: datetime) -> bool:
    start_h, start_m = map(int, SESSION_START_UTC.split(":"))
    end_h, end_m = map(int, SESSION_END_UTC.split(":"))
    current = now.hour * 60 + now.minute
    start, end = start_h * 60 + start_m, end_h * 60 + end_m
    if start == end:
        return True
    if start < end:
        return start <= current < end
    return current >= start or current < end


def validate_config() -> None:
    if ACCOUNT_EQUITY <= 0:
        raise ValueError("ACCOUNT_EQUITY must be positive")
    if not 0 < RISK_PER_TRADE <= 0.01:
        raise ValueError("RISK_PER_TRADE must be in (0, 1%] for this conservative bot")
    if not 0 < DAILY_LOSS_LIMIT <= 0.03:
        raise ValueError("DAILY_LOSS_LIMIT must be in (0, 3%]")
    if not 0.50 <= MIN_CONFIDENCE <= 0.90:
        raise ValueError("MIN_CONFIDENCE must be between 0.50 and 0.90")
    if HORIZON_BARS < 1:
        raise ValueError("HORIZON_BARS must be at least 1")
    # Validate clock strings.
    for value in (SESSION_START_UTC, SESSION_END_UTC):
        h, m = map(int, value.split(":"))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"invalid session time: {value}")


def analyze(market: str, symbol: str, raw: pd.DataFrame,
            now: Optional[datetime] = None) -> Optional[Signal]:
    now = now or utc_now()
    f = features(raw)
    if f.empty:
        return None
    row = f.iloc[-1]
    price, atr_pct = float(row["close"]), float(row["atr_pct"])
    if not math.isfinite(price) or price <= 0 or not math.isfinite(atr_pct):
        return None
    direction, confidence, reasons = score_row(row)
    stop_pct = min(0.025, max(1.5 * atr_pct, 0.002))
    target_pct = 1.5 * stop_pct
    status = "SIGNAL"
    if confidence < MIN_CONFIDENCE or direction == "NO_TRADE":
        status = "NO_TRADE"
        reasons.append("directional score below threshold")
    if not MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT:
        status = "NO_TRADE"
        reasons.append("volatility outside safety band")
    if ENFORCE_SESSION and not in_session(now):
        status = "OUTSIDE_SESSION"
        reasons.append("outside configured three-hour session")
    risk_budget = ACCOUNT_EQUITY * RISK_PER_TRADE
    # Approximate notional sizing: stop loss ~= risk budget before costs; capped at 1x equity.
    suggested_notional = min(ACCOUNT_EQUITY, risk_budget / stop_pct) if stop_pct > 0 else 0.0
    return Signal(
        market=market, symbol=symbol, timestamp_utc=now.isoformat(),
        last_price=round(price, 8), direction=direction,
        confidence_score=round(confidence, 3), confidence_is_probability=False,
        horizon_hours=HORIZON_BARS, session_utc=f"{SESSION_START_UTC}-{SESSION_END_UTC}",
        atr_pct=round(atr_pct * 100, 3), stop_distance_pct=round(stop_pct * 100, 3),
        target_distance_pct=round(target_pct * 100, 3), risk_budget=round(risk_budget, 2),
        suggested_notional=round(suggested_notional, 2), status=status, reasons=reasons
    )


def send_ntfy(title: str, message: str) -> bool:
    if not NTFY_TOPIC:
        log.info("NTFY_TOPIC unset; skipping push")
        return False
    headers = {"Title": title, "Priority": "default", "Tags": "chart_with_upwards_trend"}
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
    try:
        r = requests.post(f"{NTFY_SERVER}/{NTFY_TOPIC}", data=message.encode(),
                          headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.error("ntfy notification failed: %s", exc)
        return False


def scan(now: Optional[datetime] = None, fetcher=fetch_history) -> dict:
    now = now or utc_now()
    if ENFORCE_SESSION and not in_session(now):
        return {"generated_at_utc": now.isoformat(), "status": "OUTSIDE_SESSION", "signals": []}
    results, failures = [], []
    for market, symbol in MARKETS.items():
        raw = fetcher(symbol)
        if raw is None:
            failures.append({"market": market, "reason": "data unavailable"})
            continue
        try:
            sig = analyze(market, symbol, raw, now)
            if sig:
                results.append(asdict(sig))
        except Exception as exc:
            log.exception("Analysis error for %s", market)
            failures.append({"market": market, "reason": str(exc)})
    eligible = [s for s in results if s["status"] == "SIGNAL"]
    eligible.sort(key=lambda s: s["confidence_score"], reverse=True)
    for s in eligible[MAX_SIMULTANEOUS_SIGNALS:]:
        s["status"] = "NO_TRADE"
        s["reasons"].append("concurrent-signal cap reached")
    return {
        "generated_at_utc": now.isoformat(),
        "session_utc": f"{SESSION_START_UTC}-{SESSION_END_UTC}",
        "horizon_hours": HORIZON_BARS,
        "markets_requested": len(MARKETS),
        "markets_analyzed": len(results),
        "failures": failures,
        "disclaimer": "Signals only. Confidence is a heuristic, not a probability. No orders are placed.",
        "signals": results,
    }


def main() -> int:
    validate_config()
    now = utc_now()
    result = scan(now)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"scan-{now.date().isoformat()}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    actionable = [s for s in result.get("signals", []) if s["status"] == "SIGNAL"]
    lines = [
        f"Advanced 3-hour quant scan | {now:%Y-%m-%d %H:%M UTC}",
        f"Session: {SESSION_START_UTC}-{SESSION_END_UTC} UTC | horizon: {HORIZON_BARS} hourly bars",
        f"Markets analysed: {result.get('markets_analyzed', 0)}/{len(MARKETS)}",
        "Research alerts only. Confidence is NOT a win probability. No orders are placed.",
        ""
    ]
    if actionable:
        for s in actionable:
            lines.append(
                f'{s["market"]} ({s["symbol"]}): {s["direction"]} | price {s["last_price"]} | '
                f'score {s["confidence_score"]:.0%} (heuristic) | stop {s["stop_distance_pct"]:.2f}% | '
                f'target {s["target_distance_pct"]:.2f}% | risk budget {s["risk_budget"]:.2f} {ACCOUNT_CURRENCY} | '
                f'approx notional {s["suggested_notional"]:.2f}'
            )
    else:
        lines.append("No signals passed the filters. No trade is a valid decision.")
    send_ntfy("Advanced 3-hour quant scan", "\n".join(lines))
    log.info("Saved %s", path)
    return 0 if result.get("markets_analyzed", 0) > 0 or result.get("status") == "OUTSIDE_SESSION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
