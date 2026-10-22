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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from market_data import validate_ohlc

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
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "nii-gold-forex-bot").strip()
NTFY_SERVER = (os.getenv("NTFY_SERVER") or "https://ntfy.sh").strip().rstrip("/")
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
    volatility_ratio: float
    stop_distance_pct: float
    target_distance_pct: float
    risk_budget: float
    suggested_notional: float
    status: str
    reasons: list[str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)



SESSION_DEFINITIONS = [
    {"name": "Tokyo", "timezone": "Asia/Tokyo", "open_hour": 9, "open_minute": 0},
    {"name": "Frankfurt", "timezone": "Europe/Berlin", "open_hour": 9, "open_minute": 0},
    {"name": "London", "timezone": "Europe/London", "open_hour": 8, "open_minute": 0},
    {"name": "New York", "timezone": "America/New_York", "open_hour": 8, "open_minute": 0},
]


def upcoming_session_windows(now: datetime, horizon_hours: int = HORIZON_BARS) -> list[dict]:
    """Return the next session opening(s) and suggested GMT entry windows, DST-aware."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    events = []
    for definition in SESSION_DEFINITIONS:
        zone = ZoneInfo(definition["timezone"])
        for offset in (0, 1):
            local_day = (now + timedelta(days=offset)).astimezone(zone).date()
            if local_day.weekday() >= 5:
                continue
            local_open = datetime(
                local_day.year, local_day.month, local_day.day,
                definition["open_hour"], definition["open_minute"], tzinfo=zone
            )
            open_utc = local_open.astimezone(timezone.utc)
            minutes_until = (open_utc - now).total_seconds() / 60
            if 0 <= minutes_until <= 90:
                events.append({
                    "name": definition["name"],
                    "open_utc": open_utc,
                    "minutes_until": minutes_until,
                })
    events.sort(key=lambda item: item["open_utc"])
    if not events:
        return []
    earliest = events[0]["open_utc"]
    simultaneous = [event for event in events if event["open_utc"] == earliest]
    names = {event["name"] for event in simultaneous}
    if "New York" in names and any(
        definition["name"] == "London" and
        datetime(
            earliest.astimezone(ZoneInfo("Europe/London")).year,
            earliest.astimezone(ZoneInfo("Europe/London")).month,
            earliest.astimezone(ZoneInfo("Europe/London")).day,
            8, 0, tzinfo=ZoneInfo("Europe/London")
        ).astimezone(timezone.utc) < earliest
        for definition in SESSION_DEFINITIONS
    ):
        event_name = "London–New York overlap"
    elif len(names) > 1:
        event_name = " + ".join(sorted(names)) + " opens"
    else:
        event_name = next(iter(names))
    open_utc = earliest
    return [{
        "event": event_name,
        "session_open_gmt": open_utc.isoformat(),
        "suggested_entry_start_gmt": open_utc.isoformat(),
        "suggested_entry_end_gmt": (open_utc + timedelta(hours=horizon_hours)).isoformat(),
        "minutes_until_open": max(0, round((open_utc - now).total_seconds() / 60)),
    }]



def session_events(now: datetime) -> list[dict]:
    """Identify session opens and one-hour follow-up scans; tolerate Actions queue delays."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    events = []
    for definition in SESSION_DEFINITIONS:
        zone = ZoneInfo(definition["timezone"])
        local_now = now.astimezone(zone)
        for day_offset in (-1, 0):
            day = (local_now + timedelta(days=day_offset)).date()
            if day.weekday() >= 5:
                continue
            opened = datetime(day.year, day.month, day.day,
                              definition["open_hour"], definition["open_minute"],
                              tzinfo=zone).astimezone(timezone.utc)
            age = (now - opened).total_seconds() / 60
            if 0 <= age <= 25:
                events.append({"mode": "open", "name": definition["name"],
                               "open_gmt": opened.isoformat(), "age_minutes": round(age)})
            elif 55 <= age <= 85:
                events.append({"mode": "scan", "name": definition["name"],
                               "open_gmt": opened.isoformat(), "age_minutes": round(age)})
    # NY open is the start of the major US/European overlap while both European sessions are active.
    if any(e["name"] == "New York" and e["mode"] == "open" for e in events):
        names = {d["name"] for d in SESSION_DEFINITIONS}
        if {"London", "Frankfurt", "New York"}.issubset(names):
            ny = next(d for d in SESSION_DEFINITIONS if d["name"] == "New York")
            london = next(d for d in SESSION_DEFINITIONS if d["name"] == "London")
            frankfurt = next(d for d in SESSION_DEFINITIONS if d["name"] == "Frankfurt")
            def is_open(definition):
                z = ZoneInfo(definition["timezone"])
                local = now.astimezone(z)
                if local.weekday() >= 5:
                    return False
                opened = datetime(local.year, local.month, local.day,
                                  definition["open_hour"], definition["open_minute"], tzinfo=z)
                close_hour = 17
                closed = datetime(local.year, local.month, local.day, close_hour, 0, tzinfo=z)
                return opened <= local < closed
            if is_open(london) and is_open(frankfurt) and is_open(ny):
                events.append({"mode": "overlap", "name": "London–Frankfurt–New York overlap",
                               "open_gmt": now.isoformat(), "age_minutes": 0})
    # Deduplicate and prioritize overlap/open alerts over the one-hour follow-up scan.
    unique = {}
    for event in events:
        unique[(event["mode"], event["name"])] = event
    return list(unique.values())


def format_gmt(iso_value: str) -> str:
    value = datetime.fromisoformat(iso_value).astimezone(timezone.utc)
    return value.strftime("%Y-%m-%d %H:%M GMT")

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
            validation = validate_ohlc(df)
            if validation["errors"]:
                raise ValueError("; ".join(validation["errors"]))
            if validation["warnings"]:
                log.warning("%s data validation warnings: %s", symbol, "; ".join(validation["warnings"]))
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
    # Compare current ATR with its recent 24-bar median to detect unusual volatility.
    atr_baseline = float(f["atr_pct"].tail(24).median())
    volatility_ratio = atr_pct / atr_baseline if atr_baseline > 0 else 1.0
    # Approximate notional sizing: stop loss ~= risk budget before costs; capped at 1x equity.
    suggested_notional = min(ACCOUNT_EQUITY, risk_budget / stop_pct) if stop_pct > 0 else 0.0
    return Signal(
        market=market, symbol=symbol, timestamp_utc=now.isoformat(),
        last_price=round(price, 8), direction=direction,
        confidence_score=round(confidence, 3), confidence_is_probability=False,
        horizon_hours=HORIZON_BARS, session_utc=f"{SESSION_START_UTC}-{SESSION_END_UTC}",
        atr_pct=round(atr_pct * 100, 3), volatility_ratio=round(volatility_ratio, 2),
        stop_distance_pct=round(stop_pct * 100, 3),
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
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{NTFY_SERVER}/{NTFY_TOPIC}", data=message.encode(),
                              headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            if attempt > 1:
                log.info("ntfy notification delivered after retry %d", attempt - 1)
            return True
        except requests.RequestException as exc:
            log.error("ntfy notification attempt %d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(attempt * 2)
    return False


def scan(now: Optional[datetime] = None, fetcher=fetch_history) -> dict:
    now = now or utc_now()
    # Session-open scans are intentionally allowed before the old fixed session.
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
    windows = upcoming_session_windows(now)
    return {
        "generated_at_utc": now.isoformat(),
        "timezone": "GMT (UTC+0)",
        "horizon_hours": HORIZON_BARS,
        "session_windows": windows,
        "markets_requested": len(MARKETS),
        "markets_analyzed": len(results),
        "failures": failures,
        "disclaimer": "Signals only. Confidence is a heuristic, not a probability. No orders are placed.",
        "signals": results,
    }


def main() -> int:
    validate_config()
    now = utc_now()
    events = session_events(now)
    # Manual workflow dispatches explicitly request a real scan at any time.
    # Scheduled runs do not set this flag, so their session alerts remain unchanged.
    force_scan = os.getenv("FORCE_MARKET_SCAN", "false").strip().lower() == "true"
    if force_scan:
        events = [{"mode": "scan", "name": "Manual on-demand scan",
                   "open_gmt": now.isoformat(), "age_minutes": 0}]
    if not events:
        timestamp = format_gmt(now.isoformat())
        log.info("No session-open or one-hour follow-up event at %s; sending bot-alive test notification.", timestamp)
        # Manual runs outside a session should still prove that ntfy is configured and reachable.
        sent = send_ntfy(
            "Advanced Quant bot is running",
            f"Bot health check successful | {timestamp} GMT\n"
            "The workflow started and ntfy delivery succeeded. No market scan was run because this time is outside a configured session-open/follow-up window. "
            "To test signal generation, run during a scheduled session window."
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"scan-skipped-{now.date().isoformat()}-{now:%H%M}.json"
        path.write_text(json.dumps({
            "timestamp_gmt": timestamp,
            "status": "skipped_no_scheduled_session_event",
            "ntfy_health_notification_sent": sent,
            "message": "This run was outside the session-open and one-hour follow-up windows. A bot-alive notification was attempted; no market scan or trading signal was generated."
        }, indent=2), encoding="utf-8")
        return 0 if sent else 2

    open_events = [event for event in events if event["mode"] in ("open", "overlap")]
    scan_events = [event for event in events if event["mode"] == "scan"]

    if open_events:
        names = sorted({event["name"] for event in open_events})
        message = [
            f"Market session opened | {format_gmt(now.isoformat())}",
            "Session(s): " + ", ".join(names),
            "Times shown in GMT (UTC+0).",
        ]
        if any(event["mode"] == "overlap" for event in open_events):
            # Evaluate historical volatility at the overlap opening; do not assume overlap always means high volatility.
            overlap_result = scan(now)
            high_vol = [s for s in overlap_result["signals"] if s.get("volatility_ratio", 1) >= 1.5]
            message.append(f"Historical-volatility check: {len(high_vol)} market(s) at least 1.5x their recent 24-hour ATR baseline.")
            if high_vol:
                message.append("HIGH VOLATILITY DETECTED:")
                for s in sorted(high_vol, key=lambda item: item.get("volatility_ratio", 1), reverse=True)[:6]:
                    message.append(f'{s["market"]}: {s["volatility_ratio"]:.2f}x recent ATR | {s["direction"]} ({s["status"]})')
            else:
                message.append("No unusual volatility threshold was met; overlap alone is not a trade signal.")
        sent = send_ntfy("Market session opened", "\n".join(message))
        if not sent:
            log.error("Could not send session-open notification.")
            return 2

    if scan_events:
        result = scan(now)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"scan-{now.date().isoformat()}-{now:%H%M}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        actionable = [s for s in result.get("signals", []) if s["status"] == "SIGNAL"]
        lines = [
            f"Advanced quant follow-up scan | {format_gmt(now.isoformat())}",
            f"Session(s): {', '.join(sorted({e['name'] for e in scan_events}))}",
            f"Forecast horizon: next {HORIZON_BARS} hours | using historical hourly OHLC data",
            f"Markets analysed: {result.get('markets_analyzed', 0)}/{len(MARKETS)}",
            "Signals are heuristic research, not guaranteed predictions or win probabilities. No orders are placed.",
            f"Data quality: {len(result.get('failures', []))} market(s) failed; {result.get('markets_analyzed', 0)}/{len(MARKETS)} analysed.",
            "",
        ]
        if result.get("failures"):
            lines.append("DATA QUALITY WARNING — unavailable or invalid market data:")
            for failure in result["failures"][:6]:
                lines.append(f'- {failure.get("market", "Unknown")}: {failure.get("reason", "unknown error")}')
            if len(result["failures"]) > 6:
                lines.append(f'- Plus {len(result["failures"]) - 6} additional market failure(s); see the workflow artifact.')
            lines.append("")
        high_vol = [s for s in result["signals"] if s.get("volatility_ratio", 1) >= 1.5]
        if high_vol:
            lines.append("Unusually high volatility vs recent ATR baseline:")
            for s in sorted(high_vol, key=lambda item: item.get("volatility_ratio", 1), reverse=True)[:6]:
                lines.append(f'{s["market"]}: {s["volatility_ratio"]:.2f}x | {s["direction"]} ({s["status"]})')
            lines.append("")
        if actionable:
            for s in sorted(actionable, key=lambda item: item["confidence_score"], reverse=True):
                lines.append(
                    f'{s["market"]} ({s["symbol"]}): {s["direction"]} / '
                    f'{"BUY" if s["direction"] == "LONG" else "SELL"} | price {s["last_price"]} | '
                    f'score {s["confidence_score"]:.0%} (heuristic) | volatility {s.get("volatility_ratio", 1):.2f}x | '
                    f'stop {s["stop_distance_pct"]:.2f}% | target {s["target_distance_pct"]:.2f}% | '
                    f'risk budget {s["risk_budget"]:.2f} {ACCOUNT_CURRENCY}'
                )
                lines.append(f'  Suggested GMT review window: {format_gmt(now.isoformat())} to {format_gmt((now + timedelta(hours=HORIZON_BARS)).isoformat())}')
        elif result.get("markets_analyzed", 0) == 0:
            lines.append("DATA ERROR: zero markets were analysed; this is not a no-signal result.")
            for failure in result.get("failures", [])[:4]:
                lines.append(f'Data issue — {failure.get("market", "Unknown")}: {failure.get("reason", "unknown error")}')
        else:
            lines.append("No signals passed the filters. No trade is a valid decision.")
        if not send_ntfy("Advanced 3-hour buy/sell signals", "\n".join(lines)):
            log.error("Could not send follow-up scan notification.")
            return 2
        log.info("Saved %s", path)
        if result.get("markets_analyzed", 0) == 0:
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
