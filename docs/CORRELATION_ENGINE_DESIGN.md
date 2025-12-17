# Event Correlation Engine - Design and Architecture

> **📋 Implementation Status**: ✅ **COMPLETE**
> 
> This document describes the original design and architecture of the Event Correlation Engine. The implementation is now complete and operational.
>
> - **For practical usage and quick start**: See [`src/correlation_engine/README.md`](../src/correlation_engine/README.md)
> - **Implemented code**: [`src/correlation_engine/`](../src/correlation_engine/)
> - **Run the engine**: `python -m src.correlation_engine.server`

---

## Overview

The **Event Correlation Engine** is a component that continuously monitors events in real-time from multiple intelligence sources and automatically identifies patterns, correlations, and anomalies. Unlike on-demand correlation during analysis, the engine works continuously and proactively.

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT CORRELATION ENGINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐      ┌──────────────┐      ┌─────────────┐ │
│   │   Event      │      │  Correlation │      │   Alert     │ │
│   │  Collector   │─────►│   Rules      │─────►│  Generator  │ │
│   │              │      │   Engine     │      │             │ │
│   └──────────────┘      └──────────────┘      └─────────────┘ │
│         │                      │                      │         │
│         │                      ▼                      │         │
│         │              ┌──────────────┐               │         │
│         │              │   Pattern    │               │         │
│         └─────────────►│  Detection   │◄──────────────┘         │
│                        │   (ML/AI)    │                          │
│                        └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │  MCP    │         │  Agent  │         │ Storage │
    │ Server  │         │ (LLM)   │         │ (DB)    │
    └─────────┘         └─────────┘         └─────────┘
```

## Event Selection and Filtering

### How Events Are Selected

The Event Correlation Engine can operate in two modes:

#### 1. **Global Monitoring Mode** (Default)
- Monitors **all events** from all available sources without filtering
- Useful for discovering unexpected patterns and correlations
- Higher data volume, requires more processing power
- Best for comprehensive intelligence gathering

#### 2. **Focused Monitoring Mode** (User-Configured)
- Monitors events based on **user-defined interests** (watchlists)
- User specifies: regions, topics, keywords, event types
- More efficient, focused on specific intelligence needs
- Best for targeted monitoring of specific areas or topics

### User Configuration

Users can configure the engine with **watchlists** that define what to monitor:

```python
watchlist = {
    "name": "Ukraine Conflict Monitoring",
    "regions": [
        {
            "name": "Kharkiv Region",
            "coordinates": {"lat": 49.99, "lon": 36.23},
            "radius_km": 100
        },
        {
            "name": "Kyiv Region",
            "coordinates": {"lat": 50.45, "lon": 30.52},
            "radius_km": 50
        }
    ],
    "keywords": ["military", "strike", "explosion", "attack", "conflict"],
    "event_types": ["thermal_anomaly", "outage", "news"],
    "sources": ["satellite", "news", "cyber"],
    "priority": "high"
}
```

### Event Collection Strategy

The engine collects events based on the configured watchlist:

1. **Satellite Events (NASA FIRMS)**:
   - If watchlist has regions: Query thermal anomalies within those regions
   - If no regions: Query globally but filter by relevance (brightness threshold)
   - Time window: Last 1-6 hours (configurable)

2. **News Events (GDELT/RSS)**:
   - If watchlist has keywords: Search news matching keywords
   - If watchlist has regions: Filter by country/region codes
   - If no filters: Collect recent global news (may be limited)

3. **Cyber Events (IODA)**:
   - If watchlist has regions: Monitor connectivity for those countries
   - If no regions: Monitor global outages (may be limited to significant events)

4. **Threat Intel Events (OTX)**:
   - If watchlist has keywords: Search threat pulses matching keywords
   - If no keywords: Monitor recent high-confidence threats

5. **Telegram Events**:
   - If watchlist has keywords: Search OSINT channels for matching content
   - If no keywords: Monitor curated channels for significant events

### Example: Starting the Engine

```python
# Global monitoring (no filters)
engine = EventCorrelationEngine()
engine.start()

# Focused monitoring with watchlist
watchlist = {
    "regions": [{"lat": 49.99, "lon": 36.23, "radius_km": 100}],
    "keywords": ["military", "strike"],
    "sources": ["satellite", "news", "cyber"]
}
engine = EventCorrelationEngine(watchlist=watchlist)
engine.start()
```

### When to Use Each Mode

**Use Global Monitoring when:**
- You want to discover unexpected patterns across all sources
- You're doing exploratory intelligence gathering
- You have sufficient processing resources
- You want comprehensive coverage

**Use Focused Monitoring (with watchlist) when:**
- You're monitoring specific regions of interest (e.g., conflict zones)
- You're tracking specific topics or keywords
- You want to reduce data volume and focus on relevant events
- You have limited resources or API rate limits
- You're doing targeted intelligence gathering

**Note**: If no watchlist is provided, the engine defaults to global monitoring but may apply some basic filtering to avoid overwhelming the system with irrelevant events.

### Dynamic Watchlist Updates

Users can update watchlists at runtime:
- Add/remove regions
- Add/remove keywords
- Change priority levels
- Pause/resume specific sources

The engine will adjust its collection strategy accordingly.

## Main Components

### 1. Event Collector

**Responsibility**: Collect events from all data sources in real-time or near real-time.

**How it works**:
- Continuously monitors MCP tools (news, satellite, cyber, threat intel, telegram)
- Normalizes events into a common format
- Adds metadata (timestamp, source, location, confidence)

**Example Normalized Event**:
```python
{
    "event_id": "evt_20240115_143022_001",
    "timestamp": "2024-01-15T14:30:22Z",
    "source": "NASA FIRMS",
    "source_type": "satellite",
    "event_type": "thermal_anomaly",
    "location": {"lat": 49.99, "lon": 36.23},
    "attributes": {
        "brightness": 320.5,
        "confidence": "high",
        "frp": 15.2
    },
    "raw_data": {...}
}
```

### 2. Correlation Rules Engine

**Responsibility**: Apply correlation rules to identify connections between events.

**Rule Types**:

#### a) Temporal Correlation
```python
rule = {
    "id": "temp_001",
    "name": "Thermal anomaly + News within 1 hour",
    "type": "temporal",
    "conditions": [
        {"source_type": "satellite", "event_type": "thermal_anomaly"},
        {"source_type": "news", "keywords": ["explosion", "strike", "attack"]}
    ],
    "time_window": "1h",
    "distance_threshold_km": 10.0,
    "confidence": "high"
}
```

#### b) Geospatial Correlation
```python
rule = {
    "id": "geo_001",
    "name": "Multiple sources same location",
    "type": "geospatial",
    "conditions": [
        {"source_type": "satellite"},
        {"source_type": "news"},
        {"source_type": "cyber"}
    ],
    "distance_threshold_km": 5.0,
    "time_window": "24h",
    "min_sources": 2
}
```

#### c) Causal Correlation
```python
rule = {
    "id": "causal_001",
    "name": "Internet outage after thermal anomaly",
    "type": "causal",
    "sequence": [
        {"source_type": "satellite", "event_type": "thermal_anomaly"},
        {"source_type": "cyber", "event_type": "outage", "delay": "0-2h"}
    ],
    "location_match": True,
    "confidence": "medium"
}
```

#### d) Pattern Correlation
```python
rule = {
    "id": "pattern_001",
    "name": "Escalation pattern",
    "type": "pattern",
    "pattern": [
        {"source_type": "news", "keywords": ["tension", "military"]},
        {"source_type": "satellite", "count": ">3", "window": "6h"},
        {"source_type": "cyber", "event_type": "outage"}
    ],
    "order": "sequential",
    "time_window": "12h"
}
```

### 3. Pattern Detection

**Responsibility**: Use ML/AI to detect complex patterns that simple rules cannot capture.

**Approaches**:

#### a) Statistical Anomaly Detection
- Detects deviations from normal patterns
- Uses clustering to identify groups of similar events
- Time-series analysis for trends

#### b) LLM-based Pattern Recognition
- Uses the existing LLM agent to identify complex patterns
- Analyzes semantic context between events
- Identifies non-obvious correlations

#### c) Graph-based Correlation
- Builds an event graph
- Identifies clusters and causal paths
- Detects communities of related events

### 4. Alert Generator

**Responsibility**: Generate alerts when significant correlations are detected.

**Alert Levels**:
- **Critical**: Multiple sources confirm significant event
- **High**: Strong correlation with important implications
- **Medium**: Moderate correlation worth attention
- **Low**: Weak correlation, informational

**Alert Format**:
```python
{
    "alert_id": "alert_20240115_143500",
    "severity": "high",
    "timestamp": "2024-01-15T14:35:00Z",
    "correlation_id": "corr_001",
    "correlation_type": "geospatial-temporal",
    "description": "Thermal anomalies detected in Kharkiv region correlate with news reports of military strikes",
    "events": ["evt_001", "evt_015", "evt_022"],
    "confidence": 0.85,
    "implications": [
        "Active military engagement likely",
        "Infrastructure damage possible"
    ],
    "recommended_actions": [
        "Monitor region continuously",
        "Check connectivity status",
        "Search for additional news sources"
    ]
}
```

## Integration with Current System

### 1. Integration with MCP Server

The engine can:
- **Polling**: Make periodic queries to MCP tools
- **Webhooks**: If APIs support it, receive real-time notifications
- **Streaming**: Connect to continuous data feeds (if available)

### 2. Integration with Agent (LLM)

The engine can:
- **Enrich analyses**: Provide pre-computed correlations to the agent
- **Trigger analyses**: When detecting interesting patterns, trigger deep analyses
- **Validate hypotheses**: Use correlations to validate agent hypotheses

### 3. Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA FLOW                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Event Collector collects events from sources           │
│     ↓                                                       │
│  2. Events are normalized and stored                        │
│     ↓                                                       │
│  3. Correlation Engine applies rules                       │
│     ↓                                                       │
│  4. Pattern Detection identifies complex patterns           │
│     ↓                                                       │
│  5. Correlations are stored and evaluated                   │
│     ↓                                                       │
│  6. If significant correlation → Alert Generator           │
│     ↓                                                       │
│  7. Alert can:                                             │
│     - Notify user                                           │
│     - Trigger deep analysis from Agent                     │
│     - Log to file/system                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Technical Implementation

### Proposed Directory Structure

```
src/
├── correlation_engine/
│   ├── __init__.py
│   ├── collector.py          # Event Collector
│   ├── rules.py              # Correlation Rules Engine
│   ├── patterns.py           # Pattern Detection
│   ├── alerts.py             # Alert Generator
│   ├── storage.py            # Event storage (DB/Redis)
│   └── schemas.py            # Event/Correlation schemas
```

### Implementation Example

#### Event Collector

```python
# src/correlation_engine/collector.py
from datetime import datetime, timedelta
from typing import AsyncIterator
from src.agent.tools import MCPToolExecutor

class EventCollector:
    """Collects events from multiple sources continuously."""
    
    def __init__(
        self, 
        tool_executor: MCPToolExecutor,
        watchlist: dict | None = None
    ):
        self.tool_executor = tool_executor
        self.watchlist = watchlist  # User-defined filters
        self.collection_interval = timedelta(minutes=5)
    
    async def collect_events(self) -> AsyncIterator[dict]:
        """Collect events from all sources based on watchlist."""
        while True:
            # Collect from each source
            events = []
            
            # 1. Thermal anomalies (satellite)
            thermal_events = await self._collect_thermal_anomalies()
            events.extend(thermal_events)
            
            # 2. News events
            news_events = await self._collect_news_events()
            events.extend(news_events)
            
            # 3. Cyber events
            cyber_events = await self._collect_cyber_events()
            events.extend(cyber_events)
            
            # 4. Threat intel events
            threat_events = await self._collect_threat_events()
            events.extend(threat_events)
            
            # 5. Telegram events (if configured)
            telegram_events = await self._collect_telegram_events()
            events.extend(telegram_events)
            
            # Filter events based on watchlist if configured
            if self.watchlist:
                events = self._filter_by_watchlist(events)
            
            for event in events:
                yield event
            
            await asyncio.sleep(self.collection_interval.total_seconds())
    
    async def _collect_thermal_anomalies(self) -> list[dict]:
        """Collect recent thermal anomalies based on watchlist."""
        # If watchlist has regions, query each region
        if self.watchlist and self.watchlist.get("regions"):
            events = []
            for region in self.watchlist["regions"]:
                result = await self.tool_executor.execute(
                    "detect_thermal_anomalies",
                    {
                        "latitude": region["coordinates"]["lat"],
                        "longitude": region["coordinates"]["lon"],
                        "radius_km": region.get("radius_km", 50),
                        "hours_back": 1
                    }
                )
                events.extend(self._normalize_thermal_anomalies(result))
            return events
        else:
            # Global scan (may be limited or use default region)
            result = await self.tool_executor.execute(
                "detect_thermal_anomalies",
                {
                    "latitude": 0,  # Default or use a central point
                    "longitude": 0,
                    "radius_km": 20000,
                    "hours_back": 1
                }
            )
            return self._normalize_thermal_anomalies(result)
    
    def _normalize_thermal_anomalies(self, result: dict) -> list[dict]:
        """Normalize thermal anomaly results to event format."""
        
        events = []
        for anomaly in result.get("anomalies", []):
            events.append({
                "event_id": f"thermal_{anomaly['id']}",
                "timestamp": anomaly["acq_date"],
                "source": "NASA FIRMS",
                "source_type": "satellite",
                "event_type": "thermal_anomaly",
                "location": {
                    "lat": anomaly["latitude"],
                    "lon": anomaly["longitude"]
                },
                "attributes": {
                    "brightness": anomaly["brightness"],
                    "confidence": anomaly["confidence"],
                    "frp": anomaly.get("frp")
                }
            })
        
        return events
    
    async def _collect_news_events(self) -> list[dict]:
        """Collect news events based on watchlist keywords/regions."""
        events = []
        
        # If watchlist has keywords, search for each
        if self.watchlist and self.watchlist.get("keywords"):
            for keyword in self.watchlist["keywords"]:
                result = await self.tool_executor.execute(
                    "search_news",
                    {
                        "keywords": keyword,
                        "country_code": self._get_country_codes(),
                        "timespan": "1h"
                    }
                )
                events.extend(self._normalize_news_events(result))
        else:
            # Collect recent global news (may be limited)
            result = await self.tool_executor.execute(
                "search_news",
                {
                    "keywords": "breaking",
                    "timespan": "1h"
                }
            )
            events.extend(self._normalize_news_events(result))
        
        return events
    
    def _filter_by_watchlist(self, events: list[dict]) -> list[dict]:
        """Filter events based on watchlist criteria."""
        if not self.watchlist:
            return events
        
        filtered = []
        for event in events:
            # Check if event matches watchlist criteria
            if self._event_matches_watchlist(event):
                filtered.append(event)
        
        return filtered
    
    def _event_matches_watchlist(self, event: dict) -> bool:
        """Check if event matches watchlist criteria."""
        # Implementation would check:
        # - Source type in watchlist sources
        # - Location within watchlist regions
        # - Keywords in event content
        # - Event type in watchlist event_types
        return True  # Simplified
```

#### Correlation Rules Engine

```python
# src/correlation_engine/rules.py
from typing import List, Dict, Any
from datetime import datetime, timedelta
import math

class CorrelationRulesEngine:
    """Applies correlation rules to identify connections between events."""
    
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules
    
    def find_correlations(
        self, 
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find correlations based on rules."""
        correlations = []
        
        for rule in self.rules:
            if rule["type"] == "temporal":
                corrs = self._apply_temporal_rule(rule, events)
                correlations.extend(corrs)
            elif rule["type"] == "geospatial":
                corrs = self._apply_geospatial_rule(rule, events)
                correlations.extend(corrs)
            elif rule["type"] == "causal":
                corrs = self._apply_causal_rule(rule, events)
                correlations.extend(corrs)
            elif rule["type"] == "pattern":
                corrs = self._apply_pattern_rule(rule, events)
                correlations.extend(corrs)
        
        return correlations
    
    def _apply_temporal_rule(
        self, 
        rule: Dict[str, Any], 
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply temporal correlation rule."""
        correlations = []
        
        # Parse time window
        time_window = self._parse_time_window(rule["time_window"])
        
        # Group events by condition
        matching_events = {i: [] for i in range(len(rule["conditions"]))}
        
        for event in events:
            for i, condition in enumerate(rule["conditions"]):
                if self._event_matches_condition(event, condition):
                    matching_events[i].append(event)
        
        # Find events within time window and distance threshold
        for event1 in matching_events[0]:
            for event2 in matching_events[1]:
                time_diff = abs(
                    (datetime.fromisoformat(event1["timestamp"]) - 
                     datetime.fromisoformat(event2["timestamp"])).total_seconds()
                )
                
                if time_diff <= time_window.total_seconds():
                    # Check distance if both have location
                    if "location" in event1 and "location" in event2:
                        distance = self._calculate_distance(
                            event1["location"],
                            event2["location"]
                        )
                        
                        if distance <= rule.get("distance_threshold_km", float("inf")):
                            correlations.append({
                                "correlation_id": f"corr_{len(correlations)}",
                                "rule_id": rule["id"],
                                "correlation_type": "temporal",
                                "events": [event1["event_id"], event2["event_id"]],
                                "confidence": rule.get("confidence", "medium"),
                                "description": self._generate_description(rule, event1, event2),
                                "timestamp": datetime.utcnow().isoformat()
                            })
        
        return correlations
    
    def _calculate_distance(
        self, 
        loc1: Dict[str, float], 
        loc2: Dict[str, float]
    ) -> float:
        """Calculate distance in km between two coordinates (Haversine)."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1 = radians(loc1["lat"]), radians(loc1["lon"])
        lat2, lon2 = radians(loc2["lat"]), radians(loc2["lon"])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
```

## Use Cases

### Case 1: Military Attack Detection

**Scenario**: Multiple sources indicate military activity in a region.

**Events**:
1. Thermal anomaly detected (satellite)
2. News report of explosions (news)
3. Internet outage reported (cyber)

**Correlation**:
- **Type**: Geospatial-temporal
- **Confidence**: High
- **Implication**: Military attack likely

**Action**: Generate critical alert and trigger deep analysis from Agent.

### Case 2: Escalation Pattern

**Scenario**: Sequence of events indicating tension escalation.

**Events**:
1. News about diplomatic tension
2. Multiple thermal anomalies within 6 hours
3. Internet outage

**Correlation**:
- **Type**: Pattern (escalation)
- **Confidence**: Medium-High
- **Implication**: Conflict escalation

**Action**: Alert and recommend continuous monitoring.

### Case 3: False Positive

**Scenario**: Thermal anomaly + News, but unrelated events.

**Events**:
1. Thermal anomaly (forest fire)
2. News about politics (unrelated)

**Correlation**:
- **Type**: Temporal (but not causal)
- **Confidence**: Low
- **Validation**: LLM can verify context and discard

## Benefits

1. **Proactive Detection**: Identifies patterns before manual analysis
2. **Efficiency**: Reduces need for exploratory analyses
3. **Quick Context**: Provides pre-computed correlations to Agent
4. **Real-time Alerts**: Notifies about significant events immediately
5. **Scalability**: Can process large volumes of events

## Challenges and Considerations

1. **Data Volume**: Many events can generate many correlations
   - **Solution**: Filter by relevance and confidence

2. **False Positives**: Spurious correlations
   - **Solution**: Use LLM for contextual validation

3. **Performance**: Process correlations in real-time
   - **Solution**: Use spatial indexes (R-tree) and temporal indexes

4. **Storage**: Event and correlation history
   - **Solution**: SQLite with optimized indexes (temporal + geospatial)

5. **Rules vs ML**: Balance between explicit rules and learning
   - **Solution**: Hybrid - rules for known patterns, ML for new patterns

## Implementation Roadmap

### ✅ Completed Phases

1. **✅ Phase 1: Basic Event Collector** - COMPLETE
   - ✓ Collect events from all 5 sources (NASA FIRMS, GDELT, IODA, Telegram, OTX)
   - ✓ Normalize format using Pydantic schemas
   - ✓ Store in SQLite database with optimized indexes

2. **✅ Phase 2: Correlation Rules Engine** - COMPLETE
   - ✓ Implement 5 correlation rules (temporal, geospatial, causal patterns)
   - ✓ Distance-based correlation (Haversine)
   - ✓ Time-window based correlation

3. **✅ Phase 3: Agent Integration** - COMPLETE
   - ✓ Pre-computed correlations available to Agent via MCP
   - ✓ Dashboard API endpoints for correlation queries
   - ✓ Event storage accessible to Agent

4. **✅ Phase 4: Pattern Detection** - COMPLETE
   - ✓ Statistical anomaly detection (Z-score based)
   - ✓ Escalation pattern detection
   - ✓ LLM validation to reduce false positives (using Ollama/Gemini)

5. **✅ Phase 5: Alert System** - COMPLETE
   - ✓ Automatic severity classification (low, medium, high, critical)
   - ✓ Email notifications for high/critical alerts
   - ✓ Alert history with 90-day retention
   - ✓ HTML email templates with actionable recommendations

## Conclusion

An Event Correlation Engine would add a proactive intelligence layer to the system, enabling automatic detection of patterns and correlations between events from multiple sources. This would complement the existing LLM Agent, providing pre-computed context and triggering deep analyses when necessary.
