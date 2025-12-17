"""
SQLite storage for events and correlations.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.correlation_engine.config import ECE_DB_PATH, RETENTION_POLICY
from src.correlation_engine.schemas import Event, Correlation
from src.shared.logger import get_logger

logger = get_logger()


class EventStorage:
    """SQLite storage for events and correlations."""
    
    def __init__(self, db_path: Path | None = None):
        """Initialize storage with database path."""
        self.db_path = db_path or ECE_DB_PATH
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_schema()
        logger.info(f"Event storage initialized at {self.db_path}")
    
    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    attributes TEXT,  -- JSON
                    raw_data TEXT     -- JSON
                )
            """)
            
            # Correlations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correlations (
                    correlation_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    correlation_type TEXT NOT NULL,
                    event_ids TEXT NOT NULL,  -- JSON array
                    description TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    implications TEXT,        -- JSON array
                    recommended_actions TEXT, -- JSON array
                    rule_id TEXT
                )
            """)
            
            # Alerts table (for email/notification tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    email_sent BOOLEAN DEFAULT 0,
                    FOREIGN KEY (correlation_id) REFERENCES correlations(correlation_id)
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_source_type ON events(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_location ON events(latitude, longitude)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_timestamp ON correlations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_severity ON correlations(severity)")
            
            conn.commit()
    
    def insert_event(self, event: Event):
        """Insert an event into storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO events
                (event_id, timestamp, source, source_type, event_type, latitude, longitude, attributes, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                event.source,
                event.source_type.value,
                event.event_type,
                event.latitude,
                event.longitude,
                json.dumps(event.attributes),
                json.dumps(event.raw_data)
            ))
            conn.commit()
    
    def insert_correlation(self, correlation: Correlation):
        """Insert a correlation into storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO correlations
                (correlation_id, timestamp, correlation_type, event_ids, description, 
                 confidence, severity, implications, recommended_actions, rule_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                correlation.correlation_id,
                correlation.timestamp,
                correlation.correlation_type.value,
                json.dumps(correlation.event_ids),
                correlation.description,
                correlation.confidence,
                correlation.severity.value,
                json.dumps(correlation.implications),
                json.dumps(correlation.recommended_actions),
                correlation.rule_id
            ))
            conn.commit()
    
    def get_events(
        self,
        hours_back: float | None = None,
        source_type: str | None = None,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get events from storage."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if hours_back:
            cutoff = datetime.utcnow() - timedelta(hours=hours_back)
            query += " AND timestamp >= ?"
            params.append(cutoff.isoformat())
        
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        events = []
        for row in rows:
            events.append({
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "source_type": row["source_type"],
                "event_type": row["event_type"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "attributes": json.loads(row["attributes"]) if row["attributes"] else {},
                "raw_data": json.loads(row["raw_data"]) if row["raw_data"] else {},
            })
        
        return events
    
    def get_correlations(
        self,
        hours_back: float | None = None,
        severity: str | None = None,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get correlations from storage."""
        query = "SELECT * FROM correlations WHERE 1=1"
        params = []
        
        if hours_back:
            cutoff = datetime.utcnow() - timedelta(hours=hours_back)
            query += " AND timestamp >= ?"
            params.append(cutoff.isoformat())
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        correlations = []
        for row in rows:
            correlations.append({
                "correlation_id": row["correlation_id"],
                "timestamp": row["timestamp"],
                "correlation_type": row["correlation_type"],
                "event_ids": json.loads(row["event_ids"]),
                "description": row["description"],
                "confidence": row["confidence"],
                "severity": row["severity"],
                "implications": json.loads(row["implications"]) if row["implications"] else [],
                "recommended_actions": json.loads(row["recommended_actions"]) if row["recommended_actions"] else [],
                "rule_id": row["rule_id"],
            })
        
        return correlations
    
    def find_correlations(
        self,
        keywords: list[str] | None = None,
        region: dict[str, Any] | None = None,
        time_window: str = "24h",
        min_confidence: str = "medium"
    ) -> list[dict[str, Any]]:
        """Find correlations matching criteria (for Agent queries)."""
        # Parse time window
        hours_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
        hours = hours_map.get(time_window, 24)
        
        correlations = self.get_correlations(hours_back=hours)
        
        # Filter by confidence
        confidence_order = ["low", "medium", "high"]
        min_conf_idx = confidence_order.index(min_confidence) if min_confidence in confidence_order else 1
        
        filtered = []
        for corr in correlations:
            conf_idx = confidence_order.index(corr["confidence"]) if corr["confidence"] in confidence_order else 0
            if conf_idx >= min_conf_idx:
                # TODO: Add keyword and region filtering
                filtered.append(corr)
        
        return filtered
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        cutoff_events = datetime.utcnow() - timedelta(days=RETENTION_POLICY["events"])
        cutoff_correlations = datetime.utcnow() - timedelta(days=RETENTION_POLICY["correlations"])
        cutoff_alerts = datetime.utcnow() - timedelta(days=RETENTION_POLICY["alerts"])
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete old events
            conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_events.isoformat(),))
            
            # Delete old correlations
            conn.execute("DELETE FROM correlations WHERE timestamp < ?", (cutoff_correlations.isoformat(),))
            
            # Delete old alerts
            conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_alerts.isoformat(),))
            
            conn.commit()
        
        logger.info("Cleaned up old data from storage")
    
    def record_alert(self, correlation_id: str, email_sent: bool = False):
        """Record that an alert was generated."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (correlation_id, timestamp, email_sent)
                VALUES (?, ?, ?)
            """, (correlation_id, datetime.utcnow().isoformat(), email_sent))
            conn.commit()
