"""
MCP Server Tools Package.

All OSINT data source integrations for the MCP server.
"""

from src.mcp_server.tools.cyber import (
    check_cloudflare_radar,
    check_internet_outages,
    get_ioda_outages,
)
from src.mcp_server.tools.geo import check_nasa_firms
from src.mcp_server.tools.news import query_gdelt_events
from src.mcp_server.tools.rss import fetch_rss_feed
from src.mcp_server.tools.search import search_ddos_secrets_db, search_web
from src.mcp_server.tools.telegram import (
    get_telegram_channel_info,
    list_curated_channels,
    search_telegram_channels,
)
from src.mcp_server.tools.threat_intel import (
    get_pulse_details,
    lookup_indicator,
    search_pulses,
)

__all__ = [
    # News
    "query_gdelt_events",
    "fetch_rss_feed",
    # Search
    "search_web",
    "search_ddos_secrets_db",
    # Geo
    "check_nasa_firms",
    # Cyber
    "check_internet_outages",
    "check_cloudflare_radar",
    "get_ioda_outages",
    # Telegram
    "search_telegram_channels",
    "get_telegram_channel_info",
    "list_curated_channels",
    "lookup_indicator",
    "get_pulse_details",
    "search_pulses",
]
