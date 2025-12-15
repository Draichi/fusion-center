"""
API endpoints for the dashboard that connect to MCP server via HTTP/SSE.
"""
import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from mcp import ClientSession
from mcp.client.sse import sse_client

from src.shared.config import settings

router = APIRouter()

# Default MCP server URL
DEFAULT_MCP_SERVER_URL = f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"


def get_mcp_url(request: Request | None = None, mcp_url: str | None = None) -> str:
    """Get MCP server URL from request state or parameter."""
    if mcp_url:
        return mcp_url
    if request and hasattr(request.app.state, "mcp_url"):
        return request.app.state.mcp_url
    return DEFAULT_MCP_SERVER_URL


async def call_mcp_tool(tool_name: str, arguments: dict[str, Any], mcp_url: str | None = None) -> dict[str, Any]:
    """
    Call an MCP tool via HTTP/SSE connection.
    
    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments
        mcp_url: MCP server URL (defaults to settings)
    
    Returns:
        Tool result as dictionary
    """
    url = mcp_url or DEFAULT_MCP_SERVER_URL
    
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(tool_name, arguments=arguments)
                
                # Parse the result content
                if result.content:
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            return json.loads(content_item.text)
                
                return {"status": "error", "error_message": "No content in response"}
    except ConnectionError as e:
        return {
            "status": "error",
            "error_message": f"Could not connect to MCP server at {url}. Is it running?",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error calling MCP tool: {str(e)}",
        }


@router.get("/api/news")
async def get_latest_news(
    max_records: int = 20,
    mcp_url: str | None = None,
    request: Request = None,
) -> dict[str, Any]:
    """
    Get latest news from GDELT via MCP server.
    
    Args:
        max_records: Maximum number of articles to return (default: 20)
        mcp_url: Optional MCP server URL override
    
    Returns:
        Dictionary with news articles
    """
    url = get_mcp_url(request, mcp_url)
    result = await call_mcp_tool(
        "search_news",
        {
            "keywords": "(conflict OR military OR crisis OR attack)",
            "max_records": max_records,
            "timespan": "24h",
        },
        mcp_url=url,
    )
    
    if result.get("status") == "error":
        return {
            "status": "error",
            "articles": [],
            "count": 0,
            "error_message": result.get("error_message", "Unknown error"),
        }
    
    return {
        "status": "success",
        "articles": result.get("articles", []),
        "count": result.get("article_count", 0),
    }


@router.get("/api/thermal-anomalies")
async def get_thermal_anomalies(
    latitude: float | None = None,
    longitude: float | None = None,
    day_range: int = 7,
    mcp_url: str | None = None,
    request: Request = None,
) -> dict[str, Any]:
    """
    Get thermal anomalies from NASA FIRMS via MCP server.
    
    If no coordinates provided, returns global anomalies from key conflict zones.
    
    Args:
        latitude: Latitude (optional, defaults to multiple zones)
        longitude: Longitude (optional, defaults to multiple zones)
        day_range: Days to look back (default: 7)
        mcp_url: Optional MCP server URL override
    
    Returns:
        Dictionary with thermal anomalies
    """
    url = get_mcp_url(request, mcp_url)
    
    # If no coordinates, check multiple conflict zones in parallel
    if latitude is None or longitude is None:
        zones = [
            (50.4501, 30.5234, "Kyiv, Ukraine"),
            (31.7683, 35.2137, "Jerusalem, Israel"),
            (33.5138, 36.2765, "Damascus, Syria"),
            (35.6892, 51.3890, "Tehran, Iran"),
        ]
        
        # Fetch all zones in parallel
        tasks = [
            call_mcp_tool(
                "detect_thermal_anomalies",
                {
                    "latitude": lat,
                    "longitude": lon,
                    "day_range": day_range,
                    "radius_km": 100,
                },
                mcp_url=url,
            )
            for lat, lon, name in zones
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_anomalies = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if result.get("status") == "success":
                anomalies = result.get("anomalies", [])
                zone_name = zones[i][2]
                for anomaly in anomalies:
                    anomaly["zone_name"] = zone_name
                all_anomalies.extend(anomalies)
        
        return {
            "status": "success",
            "anomalies": all_anomalies,
            "count": len(all_anomalies),
        }
    else:
        result = await call_mcp_tool(
            "detect_thermal_anomalies",
            {
                "latitude": latitude,
                "longitude": longitude,
                "day_range": day_range,
                "radius_km": 100,
            },
            mcp_url=url,
        )
        
        if result.get("status") == "error":
            return {
                "status": "error",
                "anomalies": [],
                "count": 0,
                "error_message": result.get("error_message", "Unknown error"),
            }
        
        return {
            "status": "success",
            "anomalies": result.get("anomalies", []),
            "count": result.get("anomaly_count", 0),
        }


@router.get("/api/telegram")
async def get_telegram_posts(
    hours_back: int = 24,
    max_messages: int = 30,
    mcp_url: str | None = None,
    request: Request = None,
) -> dict[str, Any]:
    """
    Get recent posts from Telegram OSINT channels via MCP server.
    
    Args:
        hours_back: Hours to look back (default: 24)
        max_messages: Maximum messages to return (default: 30)
        mcp_url: Optional MCP server URL override
    
    Returns:
        Dictionary with Telegram messages
    """
    url = get_mcp_url(request, mcp_url)
    result = await call_mcp_tool(
        "search_telegram",
        {
            "keywords": None,
            "category": None,
            "hours_back": hours_back,
            "max_messages": max_messages,
        },
        mcp_url=url,
    )
    
    if result.get("status") == "error":
        return {
            "status": "error",
            "messages": [],
            "count": 0,
            "error_message": result.get("error_message", "Telegram not configured"),
        }
    
    return {
        "status": "success",
        "messages": result.get("messages", []),
        "count": result.get("message_count", 0),
    }


@router.get("/api/threat-intel")
async def get_threat_intel(
    limit: int = 20,
    mcp_url: str | None = None,
    request: Request = None,
) -> dict[str, Any]:
    """
    Get recent threat intelligence from AlienVault OTX via MCP server.
    
    Args:
        limit: Maximum pulses to return (default: 20)
        mcp_url: Optional MCP server URL override
    
    Returns:
        Dictionary with threat pulses
    """
    url = get_mcp_url(request, mcp_url)
    result = await call_mcp_tool(
        "search_threats",
        {
            "query": "recent threats OR malware OR APT",
            "limit": limit,
        },
        mcp_url=url,
    )
    
    if result.get("status") == "error":
        return {
            "status": "error",
            "pulses": [],
            "count": 0,
            "error_message": result.get("error_message", "OTX not configured"),
        }
    
    return {
        "status": "success",
        "pulses": result.get("pulses", []),
        "count": result.get("pulse_count", 0),
    }


@router.get("/api/dashboard")
async def get_dashboard_data(
    mcp_url: str | None = None,
    request: Request = None,
) -> dict[str, Any]:
    """
    Get all dashboard data in a single request.
    
    Args:
        mcp_url: Optional MCP server URL override
    
    Returns:
        Dictionary with all dashboard sections
    """
    try:
        url = get_mcp_url(request, mcp_url)
        
        # Fetch all data in parallel
        news_task = get_latest_news(max_records=20, mcp_url=url, request=request)
        anomalies_task = get_thermal_anomalies(day_range=7, mcp_url=url, request=request)
        telegram_task = get_telegram_posts(hours_back=24, max_messages=30, mcp_url=url, request=request)
        threat_task = get_threat_intel(limit=20, mcp_url=url, request=request)
        
        news_result, anomalies_result, telegram_result, threat_result = await asyncio.gather(
            news_task,
            anomalies_task,
            telegram_task,
            threat_task,
            return_exceptions=True,
        )
        
        # Handle exceptions
        if isinstance(news_result, Exception):
            news_result = {"status": "error", "articles": [], "count": 0, "error": str(news_result)}
        if isinstance(anomalies_result, Exception):
            anomalies_result = {"status": "error", "anomalies": [], "count": 0, "error": str(anomalies_result)}
        if isinstance(telegram_result, Exception):
            telegram_result = {"status": "error", "messages": [], "count": 0, "error": str(telegram_result)}
        if isinstance(threat_result, Exception):
            threat_result = {"status": "error", "pulses": [], "count": 0, "error": str(threat_result)}
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "mcp_server_url": url,
            "news": news_result,
            "thermal_anomalies": anomalies_result,
            "telegram": telegram_result,
            "threat_intel": threat_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")

