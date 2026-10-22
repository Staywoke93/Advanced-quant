"""Market OHLC validation helpers; these checks assess structure, not forecast accuracy."""
from __future__ import annotations
import math
import pandas as pd

REQUIRED_OHLC = ("open", "high", "low", "close")

def validate_ohlc(df: pd.DataFrame) -> dict:
    """Return structural errors and cautionary warnings for an OHLC frame.

    Price freshness is not inferred here: instrument-aware calendars are needed
    because FX/futures close on weekends while crypto trades continuously.
    """
    errors, warnings = [], []
    if df is None or df.empty:
        return {"errors": ["empty OHLC data"], "warnings": warnings}
    missing = [c for c in REQUIRED_OHLC if c not in df.columns]
    if missing:
        return {"errors": [f"missing OHLC columns: {missing}"], "warnings": warnings}
    values = df.loc[:, list(REQUIRED_OHLC)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        errors.append("OHLC contains missing or non-numeric values")
    finite = values.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    if not finite.all().all():
        errors.append("OHLC contains non-finite values")
    if (values <= 0).any().any():
        errors.append("OHLC contains non-positive prices")
    if not df.index.is_monotonic_increasing:
        errors.append("timestamps are not chronological")
    if df.index.has_duplicates:
        errors.append("duplicate timestamps")
    if not values.isna().any().any():
        bad_high = values["high"] < values[["open", "close", "low"]].max(axis=1)
        bad_low = values["low"] > values[["open", "close", "high"]].min(axis=1)
        if bad_high.any():
            errors.append(f"{int(bad_high.sum())} candle(s) have high below OHLC component")
        if bad_low.any():
            errors.append(f"{int(bad_low.sum())} candle(s) have low above OHLC component")
        jumps = int((values["close"].pct_change().abs() > 0.20).sum())
        if jumps:
            warnings.append(f"{jumps} close-to-close move(s) exceed 20%; verify provider data")
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            warnings.append("timestamps are timezone-naive")
        if len(df.index) > 1:
            deltas = df.index.to_series().diff().dropna()
            if len(deltas) and deltas.median() > pd.Timedelta(0):
                gaps = int((deltas > deltas.median() * 2.5).sum())
                if gaps:
                    warnings.append(f"{gaps} timestamp gap(s) exceed 2.5x median spacing; closures may explain them")
    return {"errors": errors, "warnings": warnings}
