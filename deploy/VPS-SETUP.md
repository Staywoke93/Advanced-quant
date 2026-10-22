# Ubuntu VPS deployment

This deployment runs the existing signals-only scanner as a supervised background
service. It scans immediately at startup and then at five minutes past each UTC
hour, matching the project's hourly market data. It does **not** place orders.
Hourly data and Yahoo Finance availability mean this is not tick-by-tick or
real-time monitoring.

## 1. Create a small Ubuntu VPS

Use a supported Ubuntu LTS image from a reputable provider. SSH into the server.
Keep SSH access restricted to your own IP where practical, use SSH keys, and apply
security updates. Do not expose any bot port publicly; this service needs no
inbound web port.

## 2. Install system packages and create a dedicated user

Run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git
sudo useradd --system --create-home --home-dir /opt/advanced-quant --shell /usr/sbin/nologin advancedquant
sudo chown advancedquant:advancedquant /opt/advanced-quant
```

If the user or directory already exists, inspect it rather than blindly rerunning
the corresponding creation command.

## 3. Get the repository and install Python dependencies

If the directory is empty:

```bash
sudo -u advancedquant git clone https://github.com/Staywoke93/Advanced-quant.git /opt/advanced-quant
cd /opt/advanced-quant
sudo -u advancedquant python3 -m venv .venv
sudo -u advancedquant .venv/bin/pip install --upgrade pip
sudo -u advancedquant .venv/bin/pip install -r requirements.txt
sudo -u advancedquant .venv/bin/python -m unittest discover -s tests -v
```

If you already cloned the repository, back up local changes first, then pull the
desired reviewed branch/commit and rerun dependency installation and tests.

## 4. Configure notifications and runtime settings

Create a private environment file. Use a long, random, unguessable ntfy topic
that is not the repository's default public topic. Anyone who knows a public
ntfy topic can subscribe to it. If you use an ntfy access token, treat it like a
password.

```bash
sudo install -o root -g advancedquant -m 0640 /dev/null /etc/advanced-quant.env
sudo nano /etc/advanced-quant.env
```

Example contents (replace the topic with your own secret value):

```dotenv
NTFY_TOPIC=replace-with-a-long-random-topic
NTFY_SERVER=https://ntfy.sh
NTFY_TOKEN=
ACCOUNT_EQUITY=1000
ACCOUNT_CURRENCY=USD
RISK_PER_TRADE=0.0025
DAILY_LOSS_LIMIT=0.01
MAX_SIMULTANEOUS_SIGNALS=2
MIN_CONFIDENCE=0.64
MIN_ATR_PCT=0.0003
MAX_ATR_PCT=0.025
HORIZON_BARS=3
DATA_INTERVAL=1h
DATA_PERIOD=730d
ENFORCE_SESSION=false
LOG_LEVEL=INFO
OUTPUT_DIR=/opt/advanced-quant/output
REQUEST_TIMEOUT=12
```

The account equity and currency are assumptions for research sizing only. The
bot does not connect to a broker, read actual account equity/P&L, enforce the
daily loss limit against live P&L, or submit/close orders. Set equity/currency
correctly for your reporting, but do not treat suggested sizing as broker-grade
position sizing.

## 5. Install and enable the systemd service

From the repository directory:

```bash
sudo install -o root -g root -m 0644 deploy/advanced-quant.service /etc/systemd/system/advanced-quant.service
sudo systemctl daemon-reload
sudo systemctl enable --now advanced-quant.service
sudo systemctl status advanced-quant.service
```

## 6. Monitor logs and service health

```bash
sudo journalctl -u advanced-quant.service -f
sudo systemctl is-active advanced-quant.service
sudo systemctl show advanced-quant.service -p MainPID -p ActiveState -p SubState
```

Press Ctrl+C to leave the log viewer; this does not stop the service.

Restart after changing code or configuration:

```bash
sudo systemctl restart advanced-quant.service
```

Stop it:

```bash
sudo systemctl disable --now advanced-quant.service
```

Scan reports are written under `output/`. Back them up or rotate/archive them if
long-term retention is needed.

## Operating notes

- The first scan runs immediately, then scans at HH:05 UTC each hour.
- The runner catches scan exceptions and tries again at the next hourly cycle;
  inspect logs to distinguish data/API failures from successful analysis.
- It is normal for forex/futures data to be unavailable during market closures.
- This is hourly-candle research, not tick-level monitoring. Provider data may be
  delayed or revised; the code's directional score is a heuristic, not a
  calibrated probability.
- No VPS deployment is active merely because these files are in GitHub. You
  must provision a server and run the installation steps above.
- Keep GitHub credentials, SSH private keys, ntfy tokens and the environment
  file out of the repository. Avoid putting secrets in shell history or logs.
