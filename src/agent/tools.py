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


class SearchSanctionsInput(BaseModel):
    """Input for searching sanctions."""
    
    query: str = Field(description="Name, alias, or identifier to search")
    entity_type: str | None = Field(
        default=None,
        description="Filter by type: 'person', 'organization', 'vessel', 'aircraft'"
    )
    countries: list[str] | None = Field(
        default=None,
        description="Filter by countries (e.g., ['RU', 'BY'])"
    )
    limit: int = Field(default=20, description="Maximum results (1-100)")


# =============================================================================
# MCP Tool Executor
# =============================================================================


# Valid tool names that exist in the MCP server
VALID_TOOL_NAMES = {
    "search_news",
    "detect_thermal_anomalies",
    "check_connectivity",
    "check_traffic_metrics",
    "search_sanctions",
    "screen_entity",
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
    "sanctions": "search_sanctions",
    "opensanctions": "search_sanctions",
    "OpenSanctions": "search_sanctions",
    "cloudflare": "check_traffic_metrics",
    "cloudflare_radar": "check_traffic_metrics",
    # Redirect removed tool to search_news
    "search_news_by_location": "search_news",
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
    
    async def search_sanctions(
        self,
        query: str,
        entity_type: str | None = None,
        countries: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search sanctions lists."""
        return await self.execute("search_sanctions", {
            "query": query,
            "entity_type": entity_type,
            "countries": countries,
            "limit": limit,
        })


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
            SouthKorea, NorthKorea, SaudiArabia, SouthAfrica, NewZealand, etc.""",
            "parameters": SearchNewsInput.model_json_schema(),
        },
        {
            "name": "detect_thermal_anomalies",
            "description": """Detect fires, explosions, and heat signatures using NASA FIRMS satellite data.
            Thermal anomalies can indicate: active fires, industrial explosions, military strikes,
            or large-scale burning events. Includes automatic retry on timeout.""",
            "parameters": DetectThermalAnomaliesInput.model_json_schema(),
        },
        {
            "name": "check_connectivity",
            "description": """Check for internet outages and connectivity issues at COUNTRY level.
            IMPORTANT: IODA only works for whole countries, NOT cities or regions.
            - Use country_code='UA' for Ukraine (not 'Odessa')
            - Use country_code='RU' for Russia (not 'Moscow')
            
            Detects: government-imposed shutdowns, infrastructure damage from conflicts,
            cyber attacks on networks, cable cuts, or routing anomalies.""",
            "parameters": CheckConnectivityInput.model_json_schema(),
        },
        {
            "name": "check_traffic_metrics",
            "description": """Query Cloudflare Radar for internet traffic and security metrics.
            Provides insights into: traffic volume changes, DDoS attacks, routing anomalies.""",
            "parameters": CheckTrafficMetricsInput.model_json_schema(),
        },
        {
            "name": "search_sanctions",
            "description": """Search sanctions lists for individuals, organizations, vessels, or aircraft.
            NOTE: Currently returns mock data - real OpenSanctions integration pending.
            Searches OFAC SDN, EU, UK, and UN sanctions lists.""",
            "parameters": SearchSanctionsInput.model_json_schema(),
        },
    ]
