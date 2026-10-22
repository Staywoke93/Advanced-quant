# Advanced Three-Hour Quant Market Bot

Conservative **signals-only** research system for FX, a gold futures proxy, and major crypto assets. Includes ntfy alerts, a TradingView Pine Script indicator, a basic chronological backtester, unit tests, and a weekday GitHub Actions workflow.

**It does not place orders. It cannot guarantee profitable predictions or zero failures.** Treat all outputs as research alerts, not financial advice.

## Architecture and data integrations

- **Scanner:** Python + Yahoo Finance (`yfinance`) hourly OHLC data.
- **Markets:** EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF; `GC=F` gold futures proxy; BTC, ETH, SOL, BNB, XRP.
- **TradingView:** `tradingview/three_hour_signal.pine` is a chart indicator. Add it to a TradingView chart (1-hour timeframe), then create alert conditions. TradingView's free/paid plan alert availability varies. The Python scanner does not read private TradingView chart data or require a TradingView API.
- **Notifications:** optional ntfy push.
- **Automation:** GitHub Actions weekday schedule, test run, scan, artifact upload.

Yahoo symbols are data-provider symbols and may not match your broker's symbol names. Gold uses a futures proxy, not spot XAU/USD. Data availability/latency varies by instrument and interval.

## Session-open and overlap scans

The workflow schedules scans approximately 30 minutes before Tokyo, Frankfurt/London, and New York opens, including the London–New York overlap. The scanner uses IANA time zones (Asia/Tokyo, Europe/Berlin, Europe/London, America/New_York) to calculate the next relevant opening and show suggested entry-window times in **GMT/UTC**, with daylight-saving changes accounted for.

Typical reference hours are Tokyo 00:00–09:00 UTC year-round, Frankfurt around 07:00–16:00 UTC in European summer / 08:00–17:00 in winter, London around 07:00–16:00 UTC in summer / 08:00–17:00 in winter, and New York around 12:00–21:00 UTC in US summer / 13:00–22:00 in winter. Exact UTC hours can differ during the weeks when the US and Europe change clocks on different dates.

Each alert includes the upcoming session, opening time, and a suggested three-hour window from the opening. This is a planning window, not an instruction to enter blindly; confirm current prices, spread, volatility, news, and the latest completed candle. The hourly data source may delay or revise candles.

GitHub Actions is not a real-time trading server; runs can be delayed. It does not watch an open position, enforce a live daily loss limit against broker equity, or close a position at the three-hour mark.

## Risk controls in this version

- Default risk budget per signal: **0.25% of assumed account equity**.
- Default daily loss limit parameter: **1%**; provided as a policy setting, but not enforceable against actual P&L in this signals-only version.
- Maximum two actionable signals per scan.
- Minimum heuristic confidence score 0.64.
- Volatility filter based on ATR.
- Stop estimate: 1.5× ATR, with a 0.2% floor and 2.5% cap.
- Target estimate: 1.5× stop distance.
- Suggested notional is capped at assumed equity.

Each scan labels the three-hour direction as RISE, FALL, or UNCERTAIN and shows an explicit **ACTION: BUY**, **ACTION: SELL**, **DIRECTIONAL BIAS: BUY/SELL (NOT ACTIONABLE)**, or **WAIT**. The action label is shown only when the signal passes configured filters; rejected setups still show the strategy's directional bias and the reasons they were filtered out. Direction is derived from the existing EMA trend, recent momentum and RSI rules; it is a directional estimate, not an exact future price or a calibrated probability. The score is a rule-based ranking heuristic, **not a calibrated win probability**. Sizing is approximate and ignores contract/tick value, account-currency conversion, leverage, margin, spreads, fees, slippage, financing, funding and broker-specific constraints. The `DAILY_LOSS_LIMIT` is documented but not enforced from broker account data.

## Set up locally

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python bot.py
```

Set environment variables from `.env.example` in your shell. The script intentionally does not load `.env` automatically. For a local scan outside the configured session, set `ENFORCE_SESSION=false`.

Run a chronological in-sample / held-out backtest (requires internet access for Yahoo Finance):

```bash
python backtest.py --symbol EURUSD=X --period 730d --interval 1h --horizon 3 --fee-bps 2 --split 0.70
```

The report separates the earlier in-sample segment from the later held-out segment in chronological order. This is a basic holdout check, not a trained-model validation or full rolling walk-forward evaluation. Backtest results are diagnostics only. Trades may overlap; exits are close-to-close and do not model intrabar stop/target ordering, realistic spread by market, slippage, swaps/funding, or broker execution. Results do not prove future profitability. Do not treat the score as a probability or promote a signal as 85–93% accurate unless repeated out-of-sample testing with adequate sample size supports that result.

The scanner and backtester now run structural OHLC checks for missing/non-numeric/non-finite or non-positive prices, impossible candle highs/lows, duplicate/non-chronological timestamps, large close-to-close jumps, and suspicious timestamp gaps. Gaps and large jumps are warnings because market closures and real volatility can explain them. These checks cannot confirm that provider prices are correct or current; instrument-aware freshness validation and broker-feed reconciliation remain future work.

## ntfy setup

1. Install ntfy on your phone and subscribe to a long, unguessable topic.
2. In GitHub repository **Settings → Secrets and variables → Actions**, add secret `NTFY_TOPIC`.
3. Optionally add secret `NTFY_TOKEN` and variable `NTFY_SERVER`.
4. Never commit tokens or topic names to a public repository. Public ntfy topics are discoverable by anyone who knows the topic.

## GitHub Actions setup

1. Create a GitHub repository and upload all project files.
2. Enable Actions/workflows.
3. Add ntfy secrets.
4. Run **Actions → Advanced three-hour quant scan → Run workflow**.
5. Review the logs and download the JSON artifact.

The scheduled workflow runs at multiple pre-open times through the weekday and at 23:30 UTC Sunday-Thursday for the next Tokyo session. Scheduled workflows can be delayed or disabled for inactive public repositories. Use workflow dispatch for a manual test.

## TradingView setup

1. Open a chart for an instrument your TradingView plan supports.
2. Select the **1-hour** timeframe.
3. Open Pine Editor, paste `tradingview/three_hour_signal.pine`, save and add it to the chart.
4. Create an alert and choose `3H LONG research signal` or `3H SHORT research signal`.
5. Treat alerts as research. The TradingView indicator is a related but not identical implementation to the Python scanner. It does not send ntfy notifications by itself; to bridge TradingView alerts into ntfy, use a secure webhook relay you control or an automation service, and protect the endpoint with a secret. Do not expose tokens in Pine source or public repositories.

## Fixed-expiry / binary options note

The strategy can supply directional research signals for a nominal three-hour horizon, but binary options have platform-specific payouts, expiry definitions, ties/refunds, and settlement rules. The included backtester does **not** calculate binary-options expected value. Before evaluating binary-style products, the payout percentage and exact settlement convention must be modeled; a directionally correct signal can still lose money if payout economics or expiry timing differ.

## Safety checklist before any financial reliance

- Verify symbol mapping and candle timestamps against your chosen platform.
- Run tests and inspect all provider failures.
- Collect paper-trading results for multiple market regimes.
- Validate signal score calibration on held-out data.
- Model realistic costs, expiry mechanics, correlations, and daily drawdown.
- Do not increase size to recover losses; maintain a manual kill switch.
- Do not enable live execution without broker-specific risk controls and monitoring.

## Configuration

See `.env.example`. All session times are UTC. Starting equity defaults to 1000 **USD**; if your 1000 is in Ghana cedis, set `ACCOUNT_CURRENCY=GHS` and `ACCOUNT_EQUITY=1000`. The current notional estimate assumes account and quoted asset currencies are comparable, so it should not be used as broker-grade order sizing.

## Limitations

This is a small, transparent baseline rather than a fully validated institutional quant system. It lacks news/event filters, a trained predictive model, walk-forward optimizer, calibrated probability estimates, broker feed reconciliation, portfolio correlation constraints, actual account P&L integration, and live execution management. It is designed to be improved through measured validation, not by assuming a prediction is correct.
