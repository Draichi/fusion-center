"""
Project Overwatch - MCP Server for OSINT and Geopolitical Intelligence.

This is the main entry point for the MCP server. It initializes the FastMCP
server, loads environment variables, and registers all available tools.

Usage:
    # Run with HTTP/SSE transport (for custom agents)
    python -m src.mcp_server.server --transport sse --port 8080

    # Run with stdio transport
    python -m src.mcp_server.server --transport stdio

    # Or using uv
    uv run python -m src.mcp_server.server --transport sse
"""

import argparse
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import tools
from src.mcp_server.tools.cyber import check_cloudflare_radar, check_internet_outages, get_ioda_outages
from src.mcp_server.tools.geo import check_nasa_firms
from src.mcp_server.tools.news import query_gdelt_events
from src.mcp_server.tools.sanctions import check_entity_sanctions, get_sanctions_info
from src.mcp_server.tools.telegram import (
    search_telegram_channels,
    get_telegram_channel_info,
    list_curated_channels,
)

# Import shared modules
from src.shared.config import settings
from src.shared.logger import (
    get_logger,
    log_config_status,
    log_startup_banner,
    log_tool_call,
    log_tools_table,
)

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger()

# Initialize FastMCP server
mcp = FastMCP(name=settings.mcp_server_name)


# =============================================================================
# GDELT News Tools
# =============================================================================


@mcp.tool()
async def search_news(
    keywords: str,
    source_country: str | None = None,
    max_records: int = 50,
    timespan: str = "7d",
) -> dict[str, Any]:
    """
    Search for news articles and events using the GDELT database.

    GDELT monitors news from around the world in 100+ languages. Use this tool to:
    - Find breaking news about conflicts, protests, or political events
    - Track media coverage of specific countries or regions
    - Monitor news about military activities or humanitarian crises
    - Research geopolitical developments and international relations

    Args:
        keywords: Search terms (e.g., "military strike", "protest", "sanctions").
                  Supports boolean operators: AND, OR (must be in parentheses), NOT.
                  Examples: "(Odessa OR Odesa) AND missile", "Ukraine AND (strike OR attack)"
        source_country: Optional country name to filter by news source origin.
                        Use GDELT format: Ukraine, Russia, China, Iran, Israel, US, UK,
                        Germany, France, etc. Multi-word names without spaces:
                        SouthKorea, NorthKorea, SaudiArabia, SouthAfrica, NewZealand.
        max_records: Maximum articles to return (1-250). Default: 50.
        timespan: Time range to search. Format: number + unit.
                  Examples: "7d" (7 days), "24h" (24 hours). Default: "7d".

    Returns:
        Dictionary with articles containing URLs, titles, dates, and sources.
    """
    log_tool_call(
        "search_news",
        keywords=keywords,
        source_country=source_country,
        max_records=max_records,
        timespan=timespan,
    )

    result = await query_gdelt_events(
        keywords=keywords,
        source_country=source_country,
        max_records=max_records,
        timespan=timespan,
    )

    logger.result_summary(
        tool_name="search_news",
        status=result.get("status", "unknown"),
        count=result.get("article_count", 0),
        details={"keywords": keywords, "timespan": timespan},
    )

    return result


# =============================================================================
# NASA FIRMS Satellite Tools
# =============================================================================


@mcp.tool()
async def detect_thermal_anomalies(
    latitude: float,
    longitude: float,
    day_range: int = 7,
    radius_km: int = 50,
) -> dict[str, Any]:
    """
    Detect thermal anomalies (fires, explosions) using NASA satellite data.

    NASA FIRMS provides near real-time active fire data from satellite observations.
    Thermal anomalies can indicate:
    - Active fires or wildfires
    - Industrial explosions or accidents
    - Military strikes or bombardments
    - Large-scale burning events

    **Note:** Requires NASA_FIRMS_API_KEY environment variable to be set.
    Get your free key at: https://firms.modaps.eosdis.nasa.gov/api/area/

    Args:
        latitude: Latitude of the center point (-90 to 90).
        longitude: Longitude of the center point (-180 to 180).
        day_range: Days to look back (1-10). Default: 7 days.
        radius_km: Search radius in kilometers (1-100). Default: 50km.

    Returns:
        Dictionary with detected thermal anomalies including coordinates,
        brightness, confidence levels, and timestamps.
    """
    log_tool_call(
        "detect_thermal_anomalies",
        latitude=latitude,
        longitude=longitude,
        day_range=day_range,
        radius_km=radius_km,
    )

    result = await check_nasa_firms(
        latitude=latitude,
        longitude=longitude,
        day_range=day_range,
        radius_km=radius_km,
    )

    logger.result_summary(
        tool_name="detect_thermal_anomalies",
        status=result.get("status", "unknown"),
        count=result.get("anomaly_count", 0),
        details={
            "location": f"({latitude}, {longitude})",
            "day_range": f"{day_range} days",
            "radius": f"{radius_km}km",
        },
    )

    return result


# =============================================================================
# Internet Infrastructure Tools
# =============================================================================


@mcp.tool()
async def check_connectivity(
    country_code: str | None = None,
    region_name: str | None = None,
    hours_back: int = 24,
) -> dict[str, Any]:
    """
    Check for internet outages and connectivity issues at COUNTRY level.

    **IMPORTANT**: IODA only provides country-level data, NOT city/region data.
    - Use country_code="UA" for Ukraine (not "Odessa" or "Kyiv")
    - Use country_code="RU" for Russia (not "Moscow")

    Uses IODA (Internet Outage Detection and Analysis) data to detect:
    - Government-imposed internet shutdowns
    - Infrastructure damage from conflicts or disasters
    - Cyber attacks on network infrastructure
    - Cable cuts or major routing anomalies

    Args:
        country_code: ISO 2-letter COUNTRY code (e.g., "UA", "RU", "IR", "SY").
                      Use this for whole countries only.
        region_name: Full COUNTRY name (e.g., "Ukraine", "Russia").
                     NOT for cities or regions - use whole country name.
        hours_back: Hours to look back for outages (1-168). Default: 24 hours.

    Returns:
        Dictionary with current connectivity status, recent outages,
        and BGP visibility metrics.
    """
    log_tool_call(
        "check_connectivity",
        country_code=country_code,
        region_name=region_name,
        hours_back=hours_back,
    )

    result = await check_internet_outages(
        country_code=country_code,
        region_name=region_name,
        hours_back=hours_back,
    )

    # Get outage count safely
    outage_count = len(result.get("recent_outages", []))
    status = result.get("current_status", {})

    logger.result_summary(
        tool_name="check_connectivity",
        status=result.get("status", "unknown"),
        count=outage_count,
        details={
            "region": country_code or region_name,
            "connectivity": status.get("status", "unknown") if status else "unknown",
        },
    )

    return result


@mcp.tool()
async def check_traffic_metrics(
    country_code: str,
    metric: str = "traffic",
) -> dict[str, Any]:
    """
    Query Cloudflare Radar for internet traffic and security metrics.

    Provides insights into:
    - Traffic volume changes that might indicate outages
    - DDoS attack patterns
    - Routing anomalies

    Args:
        country_code: ISO 2-letter country code (e.g., "UA", "RU").
        metric: Type of metric: 'traffic', 'attacks', or 'routing'. Default: 'traffic'.

    Returns:
        Dictionary with traffic metrics and anomaly indicators.
    """
    log_tool_call(
        "check_traffic_metrics",
        country_code=country_code,
        metric=metric,
    )

    result = await check_cloudflare_radar(
        country_code=country_code,
        metric=metric,
    )

    logger.result_summary(
        tool_name="check_traffic_metrics",
        status=result.get("status", "unknown"),
        count=len(result.get("metrics", {})),
        details={"country": country_code, "metric": metric},
    )

    return result


@mcp.tool()
async def get_outages(
    entity_type: str = "country",
    entity_code: str | None = None,
    days_back: int = 7,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get detected internet outage events from IODA.

    Queries IODA's outage detection system for actual outage events.
    Unlike check_connectivity which shows current status, this returns
    a list of detected outage events with severity scores.

    IODA uses multiple detection methods:
    - BGP route visibility analysis
    - Active probing (ping measurements)
    - Network telescope (darknet traffic)
    - Google Transparency Report data

    Args:
        entity_type: Type of entity to query:
                     - 'country': Country-level (use ISO 2-letter code)
                     - 'region': Sub-country regions
                     - 'asn': Autonomous System outages
                     Leave empty for global outages.
        entity_code: Code for the entity (e.g., 'UA' for Ukraine).
                     Leave empty for all entities of that type.
        days_back: Days to look back (1-90). Default: 7.
        limit: Maximum events to return (1-100). Default: 50.

    Returns:
        Dictionary with detected outage events, scores, and durations.
    """
    log_tool_call(
        "get_outages",
        entity_type=entity_type,
        entity_code=entity_code,
        days_back=days_back,
        limit=limit,
    )

    result = await get_ioda_outages(
        entity_type=entity_type,
        entity_code=entity_code,
        days_back=days_back,
        limit=limit,
    )

    logger.result_summary(
        tool_name="get_outages",
        status=result.get("status", "unknown"),
        count=result.get("outage_count", 0),
        details={
            "entity": f"{entity_type}/{entity_code}" if entity_code else entity_type,
            "days_back": days_back,
        },
    )

    return result


# =============================================================================
# Sanctions Screening Tools
# =============================================================================


@mcp.tool()
async def search_sanctions(
    query: str,
    entity_type: str | None = None,
    countries: list[str] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search sanctions lists for matching entities.

    **⚠️ STUB IMPLEMENTATION:** Currently returns mock data.
    Full OpenSanctions integration is planned for future releases.

    Searches consolidated sanctions databases for:
    - Sanctioned individuals (politicians, oligarchs, military)
    - Sanctioned organizations (companies, military units)
    - Sanctioned vessels and aircraft
    - Associated aliases and identifiers

    Args:
        query: Name, alias, or identifier to search for.
        entity_type: Filter by type: 'person', 'organization', 'vessel', 'aircraft'.
        countries: Filter by associated countries (e.g., ["RU", "BY"]).
        limit: Maximum results (1-100). Default: 20.

    Returns:
        Dictionary with matching sanctioned entities and their details.
    """
    log_tool_call(
        "search_sanctions",
        query=query,
        entity_type=entity_type,
        countries=countries,
        limit=limit,
    )

    result = await get_sanctions_info(
        query=query,
        entity_type=entity_type,
        countries=countries,
        limit=limit,
    )

    logger.result_summary(
        tool_name="search_sanctions",
        status="stub" if result.get("is_stub") else result.get("status", "unknown"),
        count=result.get("match_count", 0),
        details={"query": query, "is_stub": result.get("is_stub", True)},
    )

    return result


@mcp.tool()
async def screen_entity(
    name: str,
    date_of_birth: str | None = None,
    nationality: str | None = None,
) -> dict[str, Any]:
    """
    Perform a compliance screening check on a specific entity.

    **⚠️ STUB IMPLEMENTATION:** Currently returns mock data.

    Quick compliance check useful for:
    - Pre-transaction screening
    - Due diligence on partners
    - Counterparty verification

    Args:
        name: Full name of the person or organization.
        date_of_birth: Optional DOB (YYYY-MM-DD) for better matching.
        nationality: Optional ISO country code for nationality.

    Returns:
        Dictionary with screening results and risk assessment.
    """
    log_tool_call(
        "screen_entity",
        name=name,
        date_of_birth=date_of_birth,
        nationality=nationality,
    )

    result = await check_entity_sanctions(
        name=name,
        date_of_birth=date_of_birth,
        nationality=nationality,
    )

    screening = result.get("screening_result", {})

    logger.result_summary(
        tool_name="screen_entity",
        status="stub" if result.get("is_stub") else result.get("status", "unknown"),
        count=screening.get("matches_found", 0),
        details={"entity": name, "risk_level": screening.get("risk_level", "unknown")},
    )

    return result


# =============================================================================
# Telegram OSINT Tools
# =============================================================================


@mcp.tool()
async def search_telegram(
    keywords: str | None = None,
    channels: list[str] | None = None,
    category: str | None = None,
    hours_back: int = 24,
    max_messages: int = 50,
) -> dict[str, Any]:
    """
    Search public Telegram channels for OSINT intelligence.

    Monitors public Telegram channels for real-time information from:
    - Conflict zones (Ukraine/Russia, Middle East)
    - Independent news sources
    - Military and defense analysis channels
    - Breaking news before mainstream media

    **Note:** Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env file.
    Get credentials at: https://my.telegram.org

    Args:
        keywords: Search terms to filter messages (case-insensitive).
                  Examples: "missile", "Kharkiv", "drone strike".
                  Leave empty for all recent messages.
        channels: Specific channel usernames (without @).
                  Examples: ["ukrainenowenglish", "ryaborov"].
                  If not provided, searches curated OSINT channels.
        category: Category of curated channels:
                  - "news": Independent news (Meduza, The Insider, Kyiv Independent)
                  - "osint_general": OSINT aggregators (Rybar, Bellingcat)
        hours_back: Hours to look back (1-168). Default: 24.
        max_messages: Max messages per channel (1-100). Default: 50.

    Returns:
        Dictionary with messages containing text, timestamps, views, and links.
    """
    log_tool_call(
        "search_telegram",
        keywords=keywords,
        channels=channels,
        category=category,
        hours_back=hours_back,
        max_messages=max_messages,
    )

    result = await search_telegram_channels(
        keywords=keywords,
        channels=channels,
        category=category,
        hours_back=hours_back,
        max_messages=max_messages,
    )

    logger.result_summary(
        tool_name="search_telegram",
        status=result.get("status", "unknown"),
        count=result.get("message_count", 0),
        details={
            "keywords": keywords,
            "category": category or "all",
            "hours_back": hours_back,
        },
    )

    return result


@mcp.tool()
async def get_channel_info(
    channel_username: str,
) -> dict[str, Any]:
    """
    Get information about a specific Telegram channel.

    Retrieves metadata about a public Telegram channel including:
    - Channel name and description
    - Subscriber count
    - Verification status

    Args:
        channel_username: The channel username (with or without @).
                          Example: "ukrainenowenglish"

    Returns:
        Dictionary with channel metadata.
    """
    log_tool_call("get_channel_info", channel_username=channel_username)

    result = await get_telegram_channel_info(channel_username=channel_username)

    status = result.get("status", "unknown")
    channel_info = result.get("channel_info", {})

    logger.result_summary(
        tool_name="get_channel_info",
        status=status,
        count=1 if status == "success" else 0,
        details={"channel": channel_username, "title": channel_info.get("title")},
    )

    return result


@mcp.tool()
def list_osint_channels() -> dict[str, Any]:
    """
    List all curated OSINT Telegram channels.

    Returns a categorized list of public Telegram channels monitored
    for OSINT purposes, covering various perspectives on geopolitical events.

    Returns:
        Dictionary with channel categories and descriptions.
    """
    log_tool_call("list_osint_channels")

    result = list_curated_channels()

    logger.result_summary(
        tool_name="list_osint_channels",
        status="success",
        count=result.get("total_channels", 0),
        details={"categories": len(result.get("categories", {}))},
    )

    return result


# =============================================================================
# Server Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Project Overwatch - OSINT MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with HTTP/SSE transport (for custom agents)
  python -m src.mcp_server.server --transport sse --port 8080

  # Start server with stdio transport
  python -m src.mcp_server.server --transport stdio

  # Custom host and port
  python -m src.mcp_server.server --transport sse --host 0.0.0.0 --port 9000
        """,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport protocol: 'stdio' for pipe-based, 'sse' for HTTP/SSE (default: sse)",
    )
    parser.add_argument(
        "--host",
        default=settings.mcp_server_host,
        help=f"Host to bind to when using SSE transport (default: {settings.mcp_server_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mcp_server_port,
        help=f"Port to listen on when using SSE transport (default: {settings.mcp_server_port})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the MCP server."""
    args = parse_args()

    # Display startup banner
    log_startup_banner("server")

    # Show registered tools
    tools_info = [
        ("search_news", "📰 News", "Search GDELT for news articles"),
        ("detect_thermal_anomalies", "🛰️ Satellite", "NASA FIRMS fire detection"),
        ("check_connectivity", "🌐 Cyber", "IODA connectivity status"),
        ("get_outages", "🌐 Cyber", "IODA detected outage events"),
        ("check_traffic_metrics", "🌐 Cyber", "Cloudflare Radar metrics"),
        ("search_telegram", "📱 Telegram", "Search OSINT Telegram channels"),
        ("get_channel_info", "📱 Telegram", "Get Telegram channel info"),
        ("list_osint_channels", "📱 Telegram", "List curated OSINT channels"),
        ("search_sanctions", "🚫 Sanctions", "Search sanctions lists (stub)"),
        ("screen_entity", "🚫 Sanctions", "Entity screening (stub)"),
    ]
    log_tools_table(tools_info)

    # Show configuration status
    configs = {
        "NASA_FIRMS_API_KEY": (
            settings.has_nasa_key,
            "Required for thermal anomaly detection",
        ),
        "TELEGRAM_API_ID/HASH": (
            settings.has_telegram_credentials,
            "Required for Telegram monitoring",
        ),
        "LOG_LEVEL": (
            True,
            f"Current: {settings.log_level}",
        ),
    }
    log_config_status(configs)

    logger.divider("Server Starting")
    logger.info(f"Server name: [bold]{mcp.name}[/bold]")
    logger.info(f"Transport: [bold]{args.transport.upper()}[/bold]")

    try:
        if args.transport == "sse":
            # Configure host and port in mcp settings (FastMCP uses settings, not run() args)
            mcp.settings.host = args.host
            mcp.settings.port = args.port
            logger.info(f"Endpoint: [bold]http://{args.host}:{args.port}/sse[/bold]")
            logger.success("MCP Server starting in HTTP/SSE mode...")
            mcp.run(transport="sse")
        else:
            logger.success("MCP Server starting in stdio mode...")
            mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass  # Silent exit on Ctrl+C
