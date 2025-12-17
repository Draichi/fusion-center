"""
Configuration for the Event Correlation Engine.
"""

from pathlib import Path
from src.shared.config import settings


# Database path
ECE_DB_PATH = Path(settings.ece_db_path)

# Polling intervals (in hours)
POLLING_INTERVALS = {
    "nasa_firms": 6,      # 4x per day
    "gdelt": 2,           # 12x per day
    "ioda": 4,            # 6x per day
    "telegram": 1,        # Hourly
    "otx": 12,            # 2x per day
}

# Retention policy (in days)
RETENTION_POLICY = {
    "events": 7,          # Keep events for 7 days
    "correlations": 30,   # Keep correlations for 30 days
    "alerts": 90,         # Keep alerts for 90 days
}

# Correlation parameters
CORRELATION_TIME_WINDOW_HOURS = 24  # Default time window for correlations
CORRELATION_DISTANCE_THRESHOLD_KM = 10.0  # Default distance threshold

# Alert thresholds
ALERT_MIN_CONFIDENCE = 0.6  # Minimum confidence to generate alert
ALERT_EMAIL_SEVERITIES = ["high", "critical"]  # Send email for these severities

# ML Pattern Detection
ENABLE_ML_VALIDATION = True  # Use LLM to validate correlations
ANOMALY_DETECTION_WINDOW_DAYS = 7  # Window for statistical anomaly detection
