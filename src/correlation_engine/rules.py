"""
Correlation Rules Engine - detects patterns and correlations between events.
"""

import math
from datetime import datetime, timedelta
from typing import Any

from src.correlation_engine.schemas import (
    Correlation,
    CorrelationRule,
    CorrelationType,
    Severity,
)
from src.shared.logger import get_logger

logger = get_logger()


class CorrelationRulesEngine:
    """Applies correlation rules to identify connections between events."""
    
    def __init__(self, rules: list[CorrelationRule] | None = None):
        """Initialize with correlation rules."""
        self.rules = rules or self._get_default_rules()
        logger.info(f"Correlation Rules Engine initialized with {len(self.rules)} rules")
    
    def _get_default_rules(self) -> list[CorrelationRule]:
        """Get default correlation rules."""
        return [
            # Temporal-Geospatial: Thermal anomaly + News
            CorrelationRule(
                rule_id="thermal_news_24h",
                name="Thermal Anomaly + News (24h)",
                correlation_type=CorrelationType.TEMPORAL_GEOSPATIAL,
                conditions=[
                    {"source_type": "satellite", "event_type": "thermal_anomaly"},
                    {"source_type": "news", "keywords": ["explosion", "strike", "attack", "fire", "military"]},
                ],
                time_window_hours=24.0,
                distance_threshold_km=10.0,
                confidence="high"
            ),
            
            # Temporal-Geospatial: Thermal anomaly + Cyber outage
            CorrelationRule(
                rule_id="thermal_outage_12h",
                name="Thermal Anomaly + Internet Outage (12h)",
                correlation_type=CorrelationType.TEMPORAL_GEOSPATIAL,
                conditions=[
                    {"source_type": "satellite", "event_type": "thermal_anomaly"},
                    {"source_type": "cyber", "event_type": "outage"},
                ],
                time_window_hours=12.0,
                distance_threshold_km=50.0,  # Broader for infrastructure
                confidence="high"
            ),
            
            # Triple correlation: Satellite + News + Cyber
            CorrelationRule(
                rule_id="triple_correlation",
                name="Multi-Source Confirmation (Sat+News+Cyber)",
                correlation_type=CorrelationType.TEMPORAL_GEOSPATIAL,
                conditions=[
                    {"source_type": "satellite"},
                    {"source_type": "news"},
                    {"source_type": "cyber"},
                ],
                time_window_hours=24.0,
                distance_threshold_km=25.0,
                confidence="high"
            ),
            
            # Telegram + News correlation
            CorrelationRule(
                rule_id="telegram_news_6h",
                name="Telegram OSINT + News Confirmation",
                correlation_type=CorrelationType.TEMPORAL,
                conditions=[
                    {"source_type": "telegram"},
                    {"source_type": "news"},
                ],
                time_window_hours=6.0,
                confidence="medium"
            ),
            
            # Threat Intel + Cyber correlation
            CorrelationRule(
                rule_id="threat_cyber_24h",
                name="Threat Intel + Cyber Activity",
                correlation_type=CorrelationType.TEMPORAL,
                conditions=[
                    {"source_type": "threat_intel"},
                    {"source_type": "cyber"},
                ],
                time_window_hours=24.0,
                confidence="medium"
            ),
        ]
    
    def find_correlations(self, events: list[dict[str, Any]]) -> list[Correlation]:
        """Find correlations in events based on rules.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of detected correlations
        """
        correlations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            rule_correlations = self._apply_rule(rule, events)
            correlations.extend(rule_correlations)
        
        logger.info(f"Found {len(correlations)} correlations from {len(events)} events")
        return correlations
    
    def _apply_rule(
        self,
        rule: CorrelationRule,
        events: list[dict[str, Any]]
    ) -> list[Correlation]:
        """Apply a single rule to events."""
        correlations = []
        
        # Group events by condition
        grouped_events: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(len(rule.conditions))
        }
        
        for event in events:
            for i, condition in enumerate(rule.conditions):
                if self._event_matches_condition(event, condition):
                    grouped_events[i].append(event)
        
        # Find correlations based on rule type
        if rule.correlation_type in [CorrelationType.TEMPORAL, CorrelationType.TEMPORAL_GEOSPATIAL]:
            correlations = self._find_temporal_correlations(rule, grouped_events)
        
        return correlations
    
    def _find_temporal_correlations(
        self,
        rule: CorrelationRule,
        grouped_events: dict[int, list[dict[str, Any]]]
    ) -> list[Correlation]:
        """Find temporal correlations."""
        correlations = []
        
        # For simplicity, check pairs (can be extended to N-way correlations)
        if len(rule.conditions) < 2:
            return correlations
        
        events_group_0 = grouped_events.get(0, [])
        events_group_1 = grouped_events.get(1, [])
        
        for event1 in events_group_0:
            for event2 in events_group_1:
                # Skip if same event
                if event1["event_id"] == event2["event_id"]:
                    continue
                
                # Check temporal constraint
                time1 = datetime.fromisoformat(event1["timestamp"])
                time2 = datetime.fromisoformat(event2["timestamp"])
                time_diff_hours = abs((time1 - time2).total_seconds()) / 3600
                
                if time_diff_hours > rule.time_window_hours:
                    continue
                
                # Check geospatial constraint if applicable
                if rule.correlation_type == CorrelationType.TEMPORAL_GEOSPATIAL:
                    if rule.distance_threshold_km:
                        if not self._within_distance(event1, event2, rule.distance_threshold_km):
                            continue
                
                # Create correlation
                correlation = self._create_correlation(rule, [event1, event2])
                correlations.append(correlation)
        
        return correlations
    
    def _event_matches_condition(
        self,
        event: dict[str, Any],
        condition: dict[str, Any]
    ) -> bool:
        """Check if event matches a condition."""
        # Check source type
        if "source_type" in condition:
            if event["source_type"] != condition["source_type"]:
                return False
        
        # Check event type
        if "event_type" in condition:
            if event.get("event_type") != condition["event_type"]:
                return False
        
        # Check keywords (for news/telegram)
        if "keywords" in condition:
            event_text = str(event.get("attributes", {})).lower()
            event_text += str(event.get("raw_data", {})).lower()
            
            if not any(keyword.lower() in event_text for keyword in condition["keywords"]):
                return False
        
        return True
    
    def _within_distance(
        self,
        event1: dict[str, Any],
        event2: dict[str, Any],
        threshold_km: float
    ) -> bool:
        """Check if two events are within distance threshold."""
        lat1, lon1 = event1.get("latitude"), event1.get("longitude")
        lat2, lon2 = event2.get("latitude"), event2.get("longitude")
        
        if None in [lat1, lon1, lat2, lon2]:
            return False  # Can't check distance without coordinates
        
        distance_km = self._haversine_distance(lat1, lon1, lat2, lon2)
        return distance_km <= threshold_km
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _create_correlation(
        self,
        rule: CorrelationRule,
        events: list[dict[str, Any]]
    ) -> Correlation:
        """Create a correlation object from matched events."""
        # Generate correlation ID
        timestamp = datetime.utcnow().isoformat()
        correlation_id = f"corr_{timestamp.replace(':', '').replace('-', '').replace('.', '')[:14]}"
        
        # Determine severity based on number of sources and confidence
        num_sources = len(set(e["source_type"] for e in events))
        if num_sources >= 3 and rule.confidence == "high":
            severity = Severity.CRITICAL
        elif num_sources >= 2 and rule.confidence == "high":
            severity = Severity.HIGH
        elif rule.confidence == "high":
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        # Generate description
        source_types = ", ".join(sorted(set(e["source_type"] for e in events)))
        description = f"{rule.name}: Correlation detected between {source_types}"
        
        # Add location info if available
        locations = [
            f"{e.get('latitude', 0):.2f}N, {e.get('longitude', 0):.2f}E" 
            for e in events 
            if e.get("latitude") and e.get("longitude")
        ]
        if locations:
            description += f" near {locations[0]}"
        
        # Generate implications
        implications = []
        if "satellite" in source_types and "news" in source_types:
            implications.append("Physical event confirmed by satellite and news sources")
        if "cyber" in source_types:
            implications.append("Infrastructure impact detected")
        if num_sources >= 3:
            implications.append("Multiple independent sources confirm event")
        
        # Recommended actions
        actions = ["Monitor region for additional events", "Review full event details in dashboard"]
        if severity in [Severity.HIGH, Severity.CRITICAL]:
            actions.insert(0, "Immediate attention recommended")
        
        return Correlation(
            correlation_id=correlation_id,
            timestamp=timestamp,
            correlation_type=rule.correlation_type,
            event_ids=[e["event_id"] for e in events],
            description=description,
            confidence=rule.confidence,
            severity=severity,
            implications=implications,
            recommended_actions=actions,
            rule_id=rule.rule_id
        )
