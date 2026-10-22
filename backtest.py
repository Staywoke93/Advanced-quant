"""Simple chronological, close-to-close backtest for the rule-based signal strategy.

This is an educational baseline, not a production-grade execution simulator. It avoids
look-ahead by generating the signal from bar t and entering at close t, then measuring
the close at t+horizon. It does not model intrabar stop/target ordering, spread, fees,
funding, slippage, or broker-specific binary-option payouts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bot import features, score_row, MIN_CONFIDENCE


def backtest_frame(df: pd.DataFrame, horizon: int = 3, fee_bps: float = 2.0,
                   confidence_threshold: float = MIN_CONFIDENCE) -> dict:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    f = features(df)
    if len(f) <= horizon:
        return {"trades": 0, "win_rate": None, "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0, "profit_factor": None}
    trades = []
    # Each trade is evaluated using future close only after signal formation.
    for i in range(len(f) - horizon):
        row = f.iloc[i]
        direction, confidence, _ = score_row(row)
        if direction == "NO_TRADE" or confidence < confidence_threshold:
            continue
        entry = float(f["close"].iloc[i])
        exit_price = float(f["close"].iloc[i + horizon])
        gross = (exit_price / entry - 1) * (1 if direction == "LONG" else -1)
        net = gross - (2 * fee_bps / 10_000)
        trades.append({"timestamp": str(f.index[i]), "direction": direction, "return": net})
    if not trades:
        return {"trades": 0, "win_rate": None, "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0, "profit_factor": None}
    returns = pd.Series([t["return"] for t in trades], dtype=float)
    # Compound each sequential trade; overlapping horizons are still possible, so this
    # curve is a diagnostic and should not be interpreted as portfolio performance.
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return {
        "trades": len(trades),
        "win_rate": round(float((returns > 0).mean()), 4),
        "average_trade_pct": round(float(returns.mean() * 100), 4),
        "total_return_pct": round(float((equity.iloc[-1] - 1) * 100), 4),
        "max_drawdown_pct": round(float(dd.min() * 100), 4),
        "profit_factor": round(float(gains / losses), 4) if losses > 0 else None,
        "fee_assumption_bps_per_side": fee_bps,
        "horizon_bars": horizon,
        "warning": "Overlapping trades; close-to-close exits; no intrabar stop/target simulation. Not evidence of future profitability."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD=X")
    parser.add_argument("--period", default="730d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--output", default="output/backtest.json")
    args = parser.parse_args()
    import yfinance as yf
    df = yf.download(args.symbol, period=args.period, interval=args.interval,
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise SystemExit("No data returned. Check the symbol, interval and provider availability.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df[["open", "high", "low", "close"]].dropna()
    result = backtest_frame(df, horizon=args.horizon, fee_bps=args.fee_bps)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"symbol": args.symbol, **result}, indent=2))
    print(json.dumps({"symbol": args.symbol, **result}, indent=2))


if __name__ == "__main__":
    main()
