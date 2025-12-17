# Event Correlation Engine

Complete OSINT event correlation system with ML and email alerts.

## 🚀 Quick Start

### 1. Configure Environment Variables

Add to `.env`:

```bash
# Email Notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.email@gmail.com
SMTP_PASSWORD=your_app_password  # https://myaccount.google.com/apppasswords
SMTP_FROM=your.email@gmail.com
ALERT_EMAIL_TO=recipient@example.com

# ECE Database (optional)
ECE_DB_PATH=/var/lib/fusion-center/ece.db
```

### 2. Run the ECE

```bash
# Continuous mode (24/7)
python -m src.correlation_engine.server

# Run once (testing)
python -m src.correlation_engine.server --once

# Specify MCP Server
python -m src.correlation_engine.server --mcp-url http://localhost:8080/sse

# Specify database
python -m src.correlation_engine.server --db-path /custom/path/ece.db
```

### 3. Access via Dashboard

ECE data is exposed via API:

- `GET /api/correlations?hours_back=24&severity=high` - Correlations
- `GET /api/events?hours_back=24&source_type=satellite` - Collected events
- `GET /api/recent-alerts?hours_back=24` - Recent alerts

## 📊 Components

### 1. Event Collector
Collects events from **5 data sources**:
- **NASA FIRMS** (satellite - thermal anomalies)
- **GDELT** (news - global news)
- **IODA** (cyber - internet outages)
- **Telegram** (OSINT - curated channels)
- **AlienVault OTX** (threat intel - threats)

### 2. Correlation Rules Engine
**5 advanced correlation rules**:
- Thermal + News (24h)
- Thermal + Outage (12h)
- Triple correlation (Sat + News + Cyber)
- Telegram + News (6h)
- Threat Intel + Cyber (24h)

### 3. Pattern Detection
- **Statistical anomaly detection** (event spikes)
- **Escalation pattern detection** (increasing trends)
- **LLM validation** (reduces false positives)

### 4. Alert Generator
- Automatic severity (low, medium, high, critical)
- Email notifications for high/critical
- Alert history storage

### 5. Storage (SQLite)
- Events (7 days retention)
- Correlations (30 days)
- Alerts (90 days)
- Optimized indexes (temporal + geospatial)

## 🔧 Advanced Configuration

### Polling Intervals

Editable in `src/correlation_engine/config.py`:

```python
POLLING_INTERVALS = {
    "nasa_firms": 6,      # 4x per day
    "gdelt": 2,           # 12x per day
    "ioda": 4,            # 6x per day
    "telegram": 1,        # Hourly
    "otx": 12,            # 2x per day
}
```

### Watchlist (Geographic Filters)

```python
from src.correlation_engine.schemas import Watchlist

watchlist = Watchlist(
    name="Ukraine Conflict",
    regions=[
        {
            "name": "Kharkiv",
            "coordinates": {"lat": 49.99, "lon": 36.23},
            "radius_km": 100
        }
    ],
    keywords=["military", "strike", "attack"],
    sources=["satellite", "news", "cyber"],
    priority="high"
)

# Pass when creating EventCollector
collector = EventCollector(watchlist=watchlist)
```

## 🤖 Agent Integration

The Agent can query pre-computed correlations:

```python
# AgentState now has:
state["pre_computed_correlations"]  # ECE correlations

# Query correlations programmatically
from src.correlation_engine.storage import EventStorage

storage = EventStorage()
correlations = storage.find_correlations(
    keywords=["military", "ukraine"],
    time_window="24h",
    min_confidence="medium"
)
```

## 📈 Dashboard

Available endpoints:

```python
# Recent correlations
GET /api/correlations?hours_back=24&severity=high&limit=50

# Collected events
GET /api/events?hours_back=24&source_type=satellite&limit=100

# Recent alerts
GET /api/recent-alerts?hours_back=24
```

## 🧪 Manual Testing

```python
import asyncio
from src.correlation_engine.server import EventCorrelationEngine

async def test():
    engine = EventCorrelationEngine()
    await engine.run_once()  # Run one cycle

asyncio.run(test())
```

## 📧 Email Templates

Emails are HTML formatted with:
- Color based on severity
- Implications and recommendations
- Dashboard link
- List of involved events

## 🔒 Security

- API keys in `.env` (never commit)
- SMTP password via Gmail App Passwords
- Local database (SQLite) no external exposure
- Rate limiting in Event Collector

## 📝 Logs

ECE uses the shared logger:

```python
from src.shared.logger import get_logger
logger = get_logger()
```

Logs include:
- Events collected per source
- Detected correlations
- LLM validations
- Generated alerts
- Collection errors

## 🚀 Deploy

See `deployment_guide.md` for complete hosting instructions for:
- VPS (DigitalOcean, Hetzner)
- Docker
- Email configuration
- Supervisor (keep running 24/7)

Estimated cost: **~$5/month** to run continuously!
