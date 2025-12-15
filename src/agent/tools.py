"""
MCP Tools Wrapper for LangGraph.

Provides LangChain-compatible tools that call the MCP server.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession
from pydantic import BaseModel, Field

from src.shared.logger import get_logger

logger = get_logger()


# =============================================================================
# Tool Input Schemas (Pydantic models for structured input)
# =============================================================================


class SearchNewsInput(BaseModel):
    """Input for searching news articles."""
    
    keywords: str = Field(
        description="Search keywords with GDELT boolean syntax. "
        "IMPORTANT: OR operators MUST be inside parentheses. "
        "Examples: '(Odessa OR Odesa) AND missile', 'Ukraine AND (strike OR attack)'. "
        "WRONG: 'Odessa OR Odesa AND missile' (OR outside parentheses will fail)"
    )
    source_country: str | None = Field(
        default=None,
        description="Country name to filter by news source. Use GDELT format: "
        "Ukraine, Russia, China, Iran, Israel, US, UK, Germany, France, etc. "
        "Multi-word names without spaces: SouthKorea, NorthKorea, SaudiArabia, SouthAfrica."
    )
    max_records: int = Field(default=50, description="Maximum articles to return (1-250)")
    timespan: str = Field(default="7d", description="Time range: '7d', '24h', '30m'")


class DetectThermalAnomaliesInput(BaseModel):
    """Input for detecting thermal anomalies."""
    
    latitude: float = Field(description="Center latitude (-90 to 90)")
    longitude: float = Field(description="Center longitude (-180 to 180)")
    day_range: int = Field(default=7, description="Days to look back (1-10)")
    radius_km: int = Field(default=50, description="Search radius in km (1-100)")


class CheckConnectivityInput(BaseModel):
    """Input for checking internet connectivity."""
    
    country_code: str | None = Field(
        default=None,
        description="ISO 2-letter COUNTRY code (e.g., 'UA', 'RU', 'IR'). "
        "IMPORTANT: IODA only works at country level, NOT regions/cities. "
        "Use 'UA' for all of Ukraine, not 'Odessa' for a specific region."
    )
    region_name: str | None = Field(
        default=None,
        description="Full COUNTRY name (e.g., 'Ukraine', 'Russia'). "
        "NOT for regions or cities - use the whole country name."
    )
    hours_back: int = Field(default=24, description="Hours to look back (1-168)")


class CheckTrafficMetricsInput(BaseModel):
    """Input for checking traffic metrics."""
    
    country_code: str = Field(description="ISO 2-letter country code")
    metric: str = Field(default="traffic", description="Metric type: 'traffic', 'attacks', 'routing'")


class CheckIoCInput(BaseModel):
    """Input for checking indicators of compromise."""
    
    indicator: str = Field(
        description="The indicator value to look up (IP, domain, hash, URL, CVE, email)"
    )
    indicator_type: str = Field(
        default="IPv4",
        description="Type of indicator: 'IPv4', 'IPv6', 'domain', 'hostname', 'URL', "
        "'FileHash-MD5', 'FileHash-SHA1', 'FileHash-SHA256', 'CVE', 'email'"
    )


class SearchThreatsInput(BaseModel):
    """Input for searching threat intelligence pulses."""
    
    query: str = Field(
        description="Search terms (e.g., 'APT28 Russia', 'ransomware Ukraine')"
    )
    limit: int = Field(default=20, description="Maximum pulses to return (1-50)")


class SearchTelegramInput(BaseModel):
    """Input for searching Telegram channels."""
    
    keywords: str | None = Field(
        default=None,
        description="Search terms to filter messages (case-insensitive). "
        "Examples: 'missile', 'Kharkiv', 'drone strike'. Leave empty for all recent messages."
    )
    channels: list[str] | None = Field(
        default=None,
        description="Specific channel usernames (without @). "
        "Examples: ['meduzalive', 'rybar']. If not provided, searches curated channels."
    )
    category: str | None = Field(
        default=None,
        description="Category of curated channels: 'news' or 'osint_general'"
    )
    hours_back: int = Field(default=24, description="Hours to look back (1-168)")
    max_messages: int = Field(default=50, description="Max messages per channel (1-100)")


class FetchRssNewsInput(BaseModel):
    """Input for fetching RSS news."""
    
    source: str = Field(
        description="RSS source identifier. Options: 'meduza', 'theinsider', 'thecradle'"
    )
    max_articles: int = Field(default=20, description="Maximum articles to return (1-50)")


class GetOutagesInput(BaseModel):
    """Input for getting IODA outage events."""
    
    entity_type: str = Field(
        default="country",
        description="Type of entity: 'country', 'region', or 'asn'"
    )
    entity_code: str | None = Field(
        default=None,
        description="Entity code (e.g., 'UA' for Ukraine). Leave empty for all entities."
    )
    days_back: int = Field(default=7, description="Days to look back (1-90)")
    limit: int = Field(default=50, description="Maximum events to return (1-100)")


class GetThreatPulseInput(BaseModel):
    """Input for getting threat pulse details."""
    
    pulse_id: str = Field(description="The unique identifier of the pulse")


class GetChannelInfoInput(BaseModel):
    """Input for getting Telegram channel info."""
    
    channel_username: str = Field(
        description="The channel username (with or without @). Example: 'ukrainenowenglish'"
    )


# =============================================================================
# MCP Tool Executor
# =============================================================================


# Valid tool names that exist in the MCP server
VALID_TOOL_NAMES = {
    "search_news",
    "fetch_rss_news",
    "detect_thermal_anomalies",
    "check_connectivity",
    "get_outages",
    "check_traffic_metrics",
    "check_ioc",
    "get_threat_pulse",
    "search_threats",
    "search_telegram",
    "get_channel_info",
    "list_osint_channels",
}

# Mapping from common LLM mistakes to actual tool names
TOOL_NAME_ALIASES = {
    "gdelt": "search_news",
    "GDELT": "search_news",
    "ioda": "check_connectivity",
    "IODA": "check_connectivity",
    "nasa_firms": "detect_thermal_anomalies",
    "NASA_FIRMS": "detect_thermal_anomalies",
    "firms": "detect_thermal_anomalies",
    "FIRMS": "detect_thermal_anomalies",
    "otx": "check_ioc",
    "OTX": "check_ioc",
    "alienvault": "check_ioc",
    "AlienVault": "check_ioc",
    "ioc": "check_ioc",
    "threat_intel": "search_threats",
    "cloudflare": "check_traffic_metrics",
    "cloudflare_radar": "check_traffic_metrics",
    # Redirect removed tool to search_news
    "search_news_by_location": "search_news",
    # Telegram aliases
    "telegram": "search_telegram",
    "Telegram": "search_telegram",
}


class MCPToolExecutor:
    """
    Executes tools on the MCP server.
    
    This class maintains a session with the MCP server and provides
    methods to execute each tool.
    """
    
    def __init__(self, session: ClientSession):
        """Initialize with an active MCP session."""
        self.session = session
    
    def _resolve_tool_name(self, tool_name: str) -> str | None:
        """
        Resolve a tool name, handling aliases and validation.
        
        Args:
            tool_name: The tool name (possibly an alias)
            
        Returns:
            The resolved valid tool name, or None if invalid
        """
        # Check if it's already a valid tool name
        if tool_name in VALID_TOOL_NAMES:
            return tool_name
        
        # Check if it's an alias
        if tool_name in TOOL_NAME_ALIASES:
            resolved = TOOL_NAME_ALIASES[tool_name]
            logger.warning(f"Tool name '{tool_name}' resolved to '{resolved}' (alias mapping)")
            return resolved
        
        return None
    
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool result as dictionary
        """
        # Validate and resolve tool name
        resolved_name = self._resolve_tool_name(tool_name)
        
        if resolved_name is None:
            error_msg = (
                f"Unknown tool '{tool_name}'. "
                f"Valid tools are: {', '.join(sorted(VALID_TOOL_NAMES))}"
            )
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        logger.tool_call(resolved_name, arguments)
        
        try:
            result = await self.session.call_tool(resolved_name, arguments=arguments)
            
            # Parse the result content
            if result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        return json.loads(content_item.text)
            
            return {"status": "success", "data": str(result)}
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def search_news(
        self,
        keywords: str,
        source_country: str | None = None,
        max_records: int = 50,
        timespan: str = "7d",
    ) -> dict[str, Any]:
        """Search GDELT for news articles."""
        return await self.execute("search_news", {
            "keywords": keywords,
            "source_country": source_country,
            "max_records": max_records,
            "timespan": timespan,
        })
    
    async def detect_thermal_anomalies(
        self,
        latitude: float,
        longitude: float,
        day_range: int = 7,
        radius_km: int = 50,
    ) -> dict[str, Any]:
        """Detect thermal anomalies via NASA FIRMS."""
        return await self.execute("detect_thermal_anomalies", {
            "latitude": latitude,
            "longitude": longitude,
            "day_range": day_range,
            "radius_km": radius_km,
        })
    
    async def check_connectivity(
        self,
        country_code: str | None = None,
        region_name: str | None = None,
        hours_back: int = 24,
    ) -> dict[str, Any]:
        """Check internet connectivity/outages."""
        return await self.execute("check_connectivity", {
            "country_code": country_code,
            "region_name": region_name,
            "hours_back": hours_back,
        })
    
    async def check_traffic_metrics(
        self,
        country_code: str,
        metric: str = "traffic",
    ) -> dict[str, Any]:
        """Check Cloudflare Radar traffic metrics."""
        return await self.execute("check_traffic_metrics", {
            "country_code": country_code,
            "metric": metric,
        })
    
    async def check_ioc(
        self,
        indicator: str,
        indicator_type: str = "IPv4",
    ) -> dict[str, Any]:
        """Look up an indicator of compromise in AlienVault OTX."""
        return await self.execute("check_ioc", {
            "indicator": indicator,
            "indicator_type": indicator_type,
        })
    
    async def search_threats(
        self,
        query: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for threat intelligence pulses in AlienVault OTX."""
        return await self.execute("search_threats", {
            "query": query,
            "limit": limit,
        })
    
    async def search_telegram(
        self,
        keywords: str | None = None,
        channels: list[str] | None = None,
        category: str | None = None,
        hours_back: int = 24,
        max_messages: int = 50,
    ) -> dict[str, Any]:
        """Search Telegram OSINT channels."""
        return await self.execute("search_telegram", {
            "keywords": keywords,
            "channels": channels,
            "category": category,
            "hours_back": hours_back,
            "max_messages": max_messages,
        })
    
    async def fetch_rss_news(
        self,
        source: str,
        max_articles: int = 20,
    ) -> dict[str, Any]:
        """Fetch latest articles from RSS feeds."""
        return await self.execute("fetch_rss_news", {
            "source": source,
            "max_articles": max_articles,
        })
    
    async def get_outages(
        self,
        entity_type: str = "country",
        entity_code: str | None = None,
        days_back: int = 7,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get detected internet outage events from IODA."""
        return await self.execute("get_outages", {
            "entity_type": entity_type,
            "entity_code": entity_code,
            "days_back": days_back,
            "limit": limit,
        })
    
    async def get_threat_pulse(
        self,
        pulse_id: str,
    ) -> dict[str, Any]:
        """Get detailed information about a specific OTX threat pulse."""
        return await self.execute("get_threat_pulse", {
            "pulse_id": pulse_id,
        })
    
    async def get_channel_info(
        self,
        channel_username: str,
    ) -> dict[str, Any]:
        """Get information about a specific Telegram channel."""
        return await self.execute("get_channel_info", {
            "channel_username": channel_username,
        })
    
    async def list_osint_channels(self) -> dict[str, Any]:
        """List all curated OSINT Telegram channels."""
        return await self.execute("list_osint_channels", {})


# =============================================================================
# Tool Definitions for LangGraph
# =============================================================================


def get_tool_definitions() -> list[dict[str, Any]]:
    """
    Get tool definitions for the LLM.
    
    Returns a list of tool schemas that can be passed to the LLM
    for function calling.
    """
    return [
        {
            "name": "search_news",
            "description": """Search GDELT for news articles about geopolitical events.
            Use this to find breaking news about conflicts, protests, military activities,
            political events, and humanitarian crises.
            
            SYNTAX RULES for keywords:
            - OR operators MUST be inside parentheses
            - CORRECT: "(Odessa OR Odesa) AND (strike OR attack)"
            - CORRECT: "Ukraine AND (missile OR drone)"
            - WRONG: "Odessa OR Odesa AND missile" (will fail)
            
            For source_country, use GDELT country names:
            Ukraine, Russia, China, Iran, Israel, US, UK, Germany, France,
            SouthKorea, NorthKorea, SaudiArabia, SouthAfrica, NewZealand, etc.
            
            TIME RANGE: timespan parameter format: "7d" (days), "24h" (hours), "30m" (minutes). Default: "7d".""",
            "parameters": SearchNewsInput.model_json_schema(),
        },
        {
            "name": "detect_thermal_anomalies",
            "description": """Detect fires, explosions, and heat signatures using NASA FIRMS satellite data.
            Thermal anomalies can indicate: active fires, industrial explosions, military strikes,
            or large-scale burning events. Includes automatic retry on timeout.
            
            TIME LIMIT: day_range parameter must be between 1-10 days (default: 7 days).""",
            "parameters": DetectThermalAnomaliesInput.model_json_schema(),
        },
        {
            "name": "check_connectivity",
            "description": """Check for internet outages and connectivity issues at COUNTRY level.
            IMPORTANT: IODA only works for whole countries, NOT cities or regions.
            - Use country_code='UA' for Ukraine (not 'Odessa')
            - Use country_code='RU' for Russia (not 'Moscow')
            
            Detects: government-imposed shutdowns, infrastructure damage from conflicts,
            cyber attacks on networks, cable cuts, or routing anomalies.
            
            TIME LIMIT: hours_back parameter must be between 1-168 hours (7 days, default: 24 hours).""",
            "parameters": CheckConnectivityInput.model_json_schema(),
        },
        {
            "name": "check_traffic_metrics",
            "description": """Query Cloudflare Radar for internet traffic and security metrics.
            Provides insights into: traffic volume changes, DDoS attacks, routing anomalies.""",
            "parameters": CheckTrafficMetricsInput.model_json_schema(),
        },
        {
            "name": "check_ioc",
            "description": """Look up an indicator of compromise (IoC) in AlienVault OTX threat intelligence.
            Query for: IP addresses (malicious servers, C2), domains (phishing, malware),
            file hashes (malware samples), URLs (malicious links), CVEs (vulnerabilities).
            Returns reputation score, threat pulse count, and associated malware families.""",
            "parameters": CheckIoCInput.model_json_schema(),
        },
        {
            "name": "search_threats",
            "description": """Search AlienVault OTX for threat intelligence pulses.
            Find threat reports about: APT groups (APT28, Lazarus), malware families (Emotet, Cobalt Strike),
            campaigns (ransomware attacks), vulnerabilities (Log4j, PrintNightmare).
            Returns pulse names, descriptions, indicator counts, and targeted countries.""",
            "parameters": SearchThreatsInput.model_json_schema(),
        },
        {
            "name": "search_telegram",
            "description": """Search public Telegram channels for OSINT intelligence in real-time.
            Monitors curated channels for breaking news, conflict updates, and analysis from:
            - Independent news: Meduza, The Insider, Kyiv Independent
            - OSINT analysis: Rybar, Bellingcat
            
            Use this for real-time information from conflict zones that may not yet be in mainstream media.
            Messages include text, timestamps, view counts, and direct links.
            
            TIME LIMIT: hours_back parameter must be between 1-168 hours (7 days, default: 24 hours).""",
            "parameters": SearchTelegramInput.model_json_schema(),
        },
        {
            "name": "fetch_rss_news",
            "description": """Fetch latest articles from independent news RSS feeds.
            Supported sources:
            - meduza: Meduza (independent Russian news)
            - theinsider: The Insider (Russian investigative journalism)
            - thecradle: The Cradle (geopolitical news covering West Asia)
            
            Use this to get latest breaking news from independent sources before they appear in GDELT.""",
            "parameters": FetchRssNewsInput.model_json_schema(),
        },
        {
            "name": "get_outages",
            "description": """Get detected internet outage events from IODA.
            Unlike check_connectivity which shows current status, this returns a list of detected
            outage events with severity scores. Useful for tracking historical outages and patterns.
            
            IODA uses multiple detection methods: BGP analysis, active probing, network telescope, Google Transparency Report.
            
            TIME LIMIT: days_back parameter must be between 1-90 days (default: 7 days).""",
            "parameters": GetOutagesInput.model_json_schema(),
        },
        {
            "name": "get_threat_pulse",
            "description": """Get detailed information about a specific OTX threat pulse.
            Use this after finding a pulse via check_ioc or search_threats to get full details
            including all indicators, targeted countries, malware families, and reference URLs.""",
            "parameters": GetThreatPulseInput.model_json_schema(),
        },
        {
            "name": "get_channel_info",
            "description": """Get information about a specific Telegram channel.
            Retrieves metadata including channel name, description, subscriber count, and verification status.
            Useful for verifying channel authenticity before using in search_telegram.""",
            "parameters": GetChannelInfoInput.model_json_schema(),
        },
        {
            "name": "list_osint_channels",
            "description": """List all curated OSINT Telegram channels.
            Returns a categorized list of public Telegram channels monitored for OSINT purposes,
            covering various perspectives on geopolitical events. No parameters required.""",
            "parameters": {},
        },
    ]
