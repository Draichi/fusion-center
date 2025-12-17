"""
Event Collector - collects events from all data sources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from src.agent.tools import MCPToolExecutor
from src.correlation_engine.config import POLLING_INTERVALS
from src.correlation_engine.schemas import Event, EventSourceType, Watchlist
from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


class EventCollector:
    """Collects events from multiple data sources."""
    
    def __init__(
        self,
        mcp_server_url: str | None = None,
        watchlist: Watchlist | None = None
    ):
        """Initialize event collector.
        
        Args:
            mcp_server_url: MCP server URL (defaults to settings)
            watchlist: Optional watchlist for filtering events
        """
        self.mcp_server_url = mcp_server_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
        self.watchlist = watchlist
        self.tool_executor: MCPToolExecutor | None = None
        
        logger.info(f"Event Collector initialized (MCP: {self.mcp_server_url})")
        if watchlist:
            logger.info(f"Using watchlist: {watchlist.name}")
    
    async def collect_all_sources(self) -> list[Event]:
        """Collect events from all sources.
        
        Returns:
            List of collected events
        """
        events: list[Event] = []
        
        # Connect to MCP server
        async with sse_client(self.mcp_server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.tool_executor = MCPToolExecutor(session)
                
                # Collect from each source in parallel
                tasks = [
                    self._collect_thermal_anomalies(),
                    self._collect_news_events(),
                    self._collect_cyber_events(),
                    self._collect_telegram_events(),
                    self._collect_threat_intel_events(),
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Flatten results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error collecting events: {result}")
                        continue
                    if isinstance(result, list):
                        events.extend(result)
        
        # Filter by watchlist if configured
        if self.watchlist and self.watchlist.enabled:
            events = self._filter_by_watchlist(events)
        
        logger.info(f"Collected {len(events)} events total")
        return events
    
    async def _collect_thermal_anomalies(self) -> list[Event]:
        """Collect thermal anomalies from NASA FIRMS."""
        if not self.tool_executor:
            return []
        
        events = []
        
        try:
            # If watchlist has regions, query each
            if self.watchlist and self.watchlist.regions:
                for region in self.watchlist.regions:
                    result = await self.tool_executor.execute(
                        "detect_thermal_anomalies",
                        {
                            "latitude": region["coordinates"]["lat"],
                            "longitude": region["coordinates"]["lon"],
                            "radius_km": region.get("radius_km", 50),
                            "day_range": 1,  # Last day
                        }
                    )
                    
                    if result.get("status") == "success":
                        events.extend(self._normalize_thermal_anomalies(result))
            else:
                # Global scan - use default location or skip
                logger.debug("No regions in watchlist, skipping thermal anomalies")
        
        except Exception as e:
            logger.error(f"Error collecting thermal anomalies: {e}")
        
        logger.debug(f"Collected {len(events)} thermal anomaly events")
        return events
    
    def _normalize_thermal_anomalies(self, result: dict[str, Any]) -> list[Event]:
        """Normalize thermal anomaly data to Event objects."""
        events = []
        
        for anomaly in result.get("anomalies", []):
            event_id = f"thermal_{anomaly.get('acq_date', '')}_{anomaly.get('latitude', 0)}_{anomaly.get('longitude', 0)}"
            event_id = event_id.replace(" ", "_").replace(":", "")
            
            events.append(Event(
                event_id=event_id,
                timestamp=anomaly.get("acq_date", datetime.utcnow().isoformat()),
                source="NASA FIRMS",
                source_type=EventSourceType.SATELLITE,
                event_type="thermal_anomaly",
                latitude=anomaly.get("latitude"),
                longitude=anomaly.get("longitude"),
                attributes={
                    "brightness": anomaly.get("brightness"),
                    "confidence": anomaly.get("confidence"),
                    "frp": anomaly.get("frp"),
                },
                raw_data=anomaly
            ))
        
        return events
    
    async def _collect_news_events(self) -> list[Event]:
        """Collect news events from GDELT."""
        if not self.tool_executor:
            return []
        
        events = []
        
        try:
            # Build keywords query
            keywords = "conflict OR military OR attack"
            if self.watchlist and self.watchlist.keywords:
                keywords = " OR ".join(self.watchlist.keywords)
            
            result = await self.tool_executor.execute(
                "search_news",
                {
                    "keywords": f"({keywords})",
                    "max_records": 50,
                    "timespan": "24h",
                }
            )
            
            if result.get("status") == "success":
                events.extend(self._normalize_news_events(result))
        
        except Exception as e:
            logger.error(f"Error collecting news events: {e}")
        
        logger.debug(f"Collected {len(events)} news events")
        return events
    
    def _normalize_news_events(self, result: dict[str, Any]) -> list[Event]:
        """Normalize news data to Event objects."""
        events = []
        
        for article in result.get("articles", []):
            event_id = f"news_{article.get('url_hash', '')}_{article.get('seendate', '')}"
            event_id = event_id.replace(" ", "_").replace(":", "")
            
            events.append(Event(
                event_id=event_id,
                timestamp=article.get("seendate", datetime.utcnow().isoformat()),
                source="GDELT",
                source_type=EventSourceType.NEWS,
                event_type="news_article",
                latitude=article.get("lat"),
                longitude=article.get("lon"),
                attributes={
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "domain": article.get("domain"),
                    "language": article.get("language"),
                },
                raw_data=article
            ))
        
        return events
    
    async def _collect_cyber_events(self) -> list[Event]:
        """Collect cyber events from IODA."""
        if not self.tool_executor:
            return []
        
        events = []
        
        try:
            result = await self.tool_executor.execute(
                "check_connectivity",
                {
                    "hours_back": 24,
                }
            )
            
            if result.get("status") == "success":
                events.extend(self._normalize_cyber_events(result))
        
        except Exception as e:
            logger.error(f"Error collecting cyber events: {e}")
        
        logger.debug(f"Collected {len(events)} cyber events")
        return events
    
    def _normalize_cyber_events(self, result: dict[str, Any]) -> list[Event]:
        """Normalize cyber/outage data to Event objects."""
        events = []
        
        # IODA returns outages or connectivity data
        for outage in result.get("outages", []):
            event_id = f"cyber_{outage.get('entity', '')}_{outage.get('from', '')}"
            event_id = event_id.replace(" ", "_").replace(":", "")
            
            events.append(Event(
                event_id=event_id,
                timestamp=outage.get("from", datetime.utcnow().isoformat()),
                source="IODA",
                source_type=EventSourceType.CYBER,
                event_type="outage",
                attributes={
                    "entity": outage.get("entity"),
                    "datasource": outage.get("datasource"),
                    "score": outage.get("score"),
                },
                raw_data=outage
            ))
        
        return events
    
    async def _collect_telegram_events(self) -> list[Event]:
        """Collect events from Telegram OSINT channels."""
        if not self.tool_executor:
            return []
        
        events = []
        
        try:
            # Collect from curated OSINT channels
            keywords = None
            if self.watchlist and self.watchlist.keywords:
                keywords = " ".join(self.watchlist.keywords)
            
            result = await self.tool_executor.execute(
                "search_telegram",
                {
                    "keywords": keywords,
                    "category": "osint_general",
                    "hours_back": 24,
                    "max_messages": 50,
                }
            )
            
            if result.get("status") == "success":
                events.extend(self._normalize_telegram_events(result))
        
        except Exception as e:
            logger.error(f"Error collecting telegram events: {e}")
        
        logger.debug(f"Collected {len(events)} telegram events")
        return events
    
    def _normalize_telegram_events(self, result: dict[str, Any]) -> list[Event]:
        """Normalize Telegram messages to Event objects."""
        events = []
        
        for message in result.get("messages", []):
            event_id = f"telegram_{message.get('channel', '')}_{message.get('date', '')}"
            event_id = event_id.replace(" ", "_").replace(":", "").replace("@", "")
            
            events.append(Event(
                event_id=event_id,
                timestamp=message.get("date", datetime.utcnow().isoformat()),
                source=f"Telegram @{message.get('channel', 'unknown')}",
                source_type=EventSourceType.TELEGRAM,
                event_type="telegram_message",
                attributes={
                    "channel": message.get("channel"),
                    "text": message.get("text"),
                    "views": message.get("views"),
                },
                raw_data=message
            ))
        
        return events
    
    async def _collect_threat_intel_events(self) -> list[Event]:
        """Collect threat intelligence from OTX."""
        if not self.tool_executor:
            return []
        
        events = []
        
        try:
            # Search for recent threats
            query = "APT OR malware OR ransomware"
            if self.watchlist and self.watchlist.keywords:
                query = " OR ".join(self.watchlist.keywords)
            
            result = await self.tool_executor.execute(
                "search_threats",
                {
                    "query": query,
                    "limit": 20,
                }
            )
            
            if result.get("status") == "success":
                events.extend(self._normalize_threat_intel_events(result))
        
        except Exception as e:
            logger.error(f"Error collecting threat intel events: {e}")
        
        logger.debug(f"Collected {len(events)} threat intel events")
        return events
    
    def _normalize_threat_intel_events(self, result: dict[str, Any]) -> list[Event]:
        """Normalize OTX threat pulses to Event objects."""
        events = []
        
        for pulse in result.get("pulses", []):
            event_id = f"threat_{pulse.get('id', '')}"
            
            events.append(Event(
                event_id=event_id,
                timestamp=pulse.get("created", datetime.utcnow().isoformat()),
                source="AlienVault OTX",
                source_type=EventSourceType.THREAT_INTEL,
                event_type="threat_pulse",
                attributes={
                    "name": pulse.get("name"),
                    "description": pulse.get("description"),
                    "tags": pulse.get("tags", []),
                    "tlp": pulse.get("TLP"),
                },
                raw_data=pulse
            ))
        
        return events
    
    def _filter_by_watchlist(self, events: list[Event]) -> list[Event]:
        """Filter events based on watchlist criteria."""
        if not self.watchlist:
            return events
        
        filtered = []
        
        for event in events:
            if self._event_matches_watchlist(event):
                filtered.append(event)
        
        logger.debug(f"Filtered {len(events)} events to {len(filtered)} using watchlist")
        return filtered
    
    def _event_matches_watchlist(self, event: Event) -> bool:
        """Check if event matches watchlist criteria."""
        if not self.watchlist:
            return True
        
        # Check source types
        if self.watchlist.sources:
            if event.source_type.value not in self.watchlist.sources:
                return False
        
        # Check event types
        if self.watchlist.event_types:
            if event.event_type not in self.watchlist.event_types:
                return False
        
        # Check geographic regions
        if self.watchlist.regions and event.latitude and event.longitude:
            in_region = False
            for region in self.watchlist.regions:
                distance = self._calculate_distance(
                    event.latitude,
                    event.longitude,
                    region["coordinates"]["lat"],
                    region["coordinates"]["lon"]
                )
                if distance <= region.get("radius_km", 50):
                    in_region = True
                    break
            
            if not in_region:
                return False
        
        # Check keywords (for news/telegram)
        if self.watchlist.keywords and event.source_type in [EventSourceType.NEWS, EventSourceType.TELEGRAM]:
            event_text = str(event.attributes).lower()
            has_keyword = any(kw.lower() in event_text for kw in self.watchlist.keywords)
            if not has_keyword:
                return False
        
        return True
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km (Haversine)."""
        import math
        
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
