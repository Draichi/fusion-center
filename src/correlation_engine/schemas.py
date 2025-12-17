"""
Event and Correlation schemas for the Event Correlation Engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventSourceType(str, Enum):
    """Types of event sources."""
    
    SATELLITE = "satellite"
    NEWS = "news"
    CYBER = "cyber"
    TELEGRAM = "telegram"
    THREAT_INTEL = "threat_intel"


class CorrelationType(str, Enum):
    """Types of correlations."""
    
    TEMPORAL = "temporal"
    GEOSPATIAL = "geospatial"
    TEMPORAL_GEOSPATIAL = "temporal-geospatial"
    CAUSAL = "causal"
    PATTERN = "pattern"


class Severity(str, Enum):
    """Alert severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Normalized event from any source."""
    
    event_id: str
    timestamp: str  # ISO format
    source: str  # e.g., "NASA FIRMS", "GDELT"
    source_type: EventSourceType
    event_type: str  # e.g., "thermal_anomaly", "news_article"
    
    # Optional location
    latitude: float | None = None
    longitude: float | None = None
    
    # Extra attributes (source-specific)
    attributes: dict[str, Any] = field(default_factory=dict)
    
    # Raw data from source
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_type": self.source_type.value,
            "event_type": self.event_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "attributes": self.attributes,
            "raw_data": self.raw_data,
        }


@dataclass
class Correlation:
    """Correlation between multiple events."""
    
    correlation_id: str
    timestamp: str  # When correlation was detected
    correlation_type: CorrelationType
    event_ids: list[str]  # Events involved
    
    # Correlation metadata
    description: str
    confidence: str  # "low", "medium", "high"
    severity: Severity
    
    # Analysis
    implications: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    
    # Rule that triggered this correlation
    rule_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "correlation_type": self.correlation_type.value,
            "event_ids": self.event_ids,
            "description": self.description,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "implications": self.implications,
            "recommended_actions": self.recommended_actions,
            "rule_id": self.rule_id,
        }


@dataclass
class CorrelationRule:
    """Rule for detecting correlations."""
    
    rule_id: str
    name: str
    correlation_type: CorrelationType
    
    # Conditions to match
    conditions: list[dict[str, Any]]
    
    # Temporal constraints
    time_window_hours: float = 24.0  # Events within this window
    
    # Geospatial constraints
    distance_threshold_km: float | None = None
    
    # Confidence level of rule
    confidence: str = "medium"  # "low", "medium", "high"
    
    # Enabled/disabled
    enabled: bool = True


@dataclass
class Watchlist:
    """Watchlist configuration for filtering events."""
    
    name: str
    
    # Geographic regions of interest
    regions: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{"name": "...", "coordinates": {"lat": ..., "lon": ...}, "radius_km": ...}]
    
    # Keywords for news/telegram filtering
    keywords: list[str] = field(default_factory=list)
    
    # Event types to monitor
    event_types: list[str] = field(default_factory=list)
    
    # Source types to include
    sources: list[str] = field(default_factory=list)
    
    # Priority (affects polling frequency)
    priority: str = "medium"  # "low", "medium", "high"
    
    # Enabled/disabled
    enabled: bool = True
