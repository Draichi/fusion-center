"""
AlienVault OTX (Open Threat Exchange) Integration Module.

This module provides tools to query AlienVault OTX for threat intelligence,
including indicators of compromise (IoCs), pulses, and reputation data.
"""

from datetime import datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from src.shared.config import settings

# AlienVault OTX API base URL
OTX_API_BASE_URL = "https://otx.alienvault.com/api/v1"


class IndicatorInfo(BaseModel):
    """Represents an indicator of compromise from OTX."""

    indicator: str = Field(description="The indicator value (IP, domain, hash, etc.)")
    indicator_type: str = Field(description="Type of indicator (IPv4, domain, hostname, FileHash-MD5, etc.)")
    pulse_count: int = Field(default=0, description="Number of pulses referencing this indicator")
    reputation: int | None = Field(default=None, description="Reputation score (if available)")
    country: str | None = Field(default=None, description="Country code for IP indicators")
    asn: str | None = Field(default=None, description="ASN for IP indicators")
    whois: str | None = Field(default=None, description="WHOIS information summary")
    malware_families: list[str] = Field(default_factory=list, description="Associated malware families")
    last_seen: str | None = Field(default=None, description="Last time this indicator was seen")


class PulseInfo(BaseModel):
    """Represents a threat intelligence pulse from OTX."""

    pulse_id: str = Field(description="Unique pulse identifier")
    name: str = Field(description="Pulse name/title")
    description: str = Field(default="", description="Pulse description")
    author_name: str = Field(default="", description="Author of the pulse")
    created: str = Field(description="Creation timestamp")
    modified: str = Field(description="Last modified timestamp")
    tags: list[str] = Field(default_factory=list, description="Associated tags")
    targeted_countries: list[str] = Field(default_factory=list, description="Targeted countries")
    malware_families: list[str] = Field(default_factory=list, description="Associated malware families")
    indicator_count: int = Field(default=0, description="Number of indicators in the pulse")
    references: list[str] = Field(default_factory=list, description="Reference URLs")


IndicatorType = Literal[
    "IPv4", "IPv6", "domain", "hostname", "URL", 
    "FileHash-MD5", "FileHash-SHA1", "FileHash-SHA256",
    "CVE", "email"
]


def _get_headers() -> dict[str, str]:
    """Get headers for OTX API requests."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "ProjectOverwatch/1.0",
    }
    if settings.otx_api_key:
        headers["X-OTX-API-KEY"] = settings.otx_api_key
    return headers


async def lookup_indicator(
    indicator: str,
    indicator_type: IndicatorType = "IPv4",
) -> dict[str, Any]:
    """
    Look up an indicator of compromise in AlienVault OTX.

    Query the OTX database for threat intelligence about a specific indicator.
    Supported indicator types:
    - IPv4/IPv6: IP addresses
    - domain/hostname: Domain names
    - URL: Full URLs
    - FileHash-MD5/SHA1/SHA256: File hashes
    - CVE: Common Vulnerabilities and Exposures IDs
    - email: Email addresses

    Args:
        indicator: The indicator value to look up (e.g., "8.8.8.8", "example.com", 
                   "44d88612fea8a8f36de82e1278abb02f").
        indicator_type: Type of indicator. Options:
                        - "IPv4": IPv4 address
                        - "IPv6": IPv6 address
                        - "domain": Domain name
                        - "hostname": Hostname
                        - "URL": Full URL
                        - "FileHash-MD5": MD5 file hash
                        - "FileHash-SHA1": SHA1 file hash
                        - "FileHash-SHA256": SHA256 file hash
                        - "CVE": CVE identifier
                        - "email": Email address

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - indicator_info: Details about the indicator including reputation and pulse count
        - pulses: List of threat intelligence pulses referencing this indicator
        - geo_info: Geolocation data (for IP addresses)
        - error_message: Error details if the query failed

    Example:
        >>> result = await lookup_indicator("8.8.8.8", "IPv4")
        >>> print(f"Pulse count: {result['indicator_info']['pulse_count']}")
    """
    if not settings.has_otx_key:
        return {
            "status": "error",
            "error_message": "OTX_API_KEY not configured. Get your free API key at https://otx.alienvault.com",
            "indicator": indicator,
            "indicator_type": indicator_type,
        }

    query_params = {
        "indicator": indicator,
        "indicator_type": indicator_type,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get general information
            general_url = f"{OTX_API_BASE_URL}/indicators/{indicator_type}/{indicator}/general"
            general_response = await client.get(general_url, headers=_get_headers())
            
            if general_response.status_code == 404:
                return {
                    "status": "success",
                    "query_params": query_params,
                    "indicator_info": {
                        "indicator": indicator,
                        "indicator_type": indicator_type,
                        "pulse_count": 0,
                        "reputation": None,
                        "message": "Indicator not found in OTX database (this may be a good sign)",
                    },
                    "pulses": [],
                    "geo_info": None,
                }
            
            if general_response.status_code != 200:
                return {
                    "status": "error",
                    "error_message": f"OTX API error: HTTP {general_response.status_code}",
                    "query_params": query_params,
                }
            
            general_data = general_response.json()
            
            # Build indicator info
            indicator_info = IndicatorInfo(
                indicator=indicator,
                indicator_type=indicator_type,
                pulse_count=general_data.get("pulse_info", {}).get("count", 0),
                reputation=general_data.get("reputation"),
                country=general_data.get("country_code"),
                asn=general_data.get("asn"),
                whois=general_data.get("whois"),
                malware_families=general_data.get("pulse_info", {}).get("malware_families", []) or [],
                last_seen=None,
            )
            
            # Extract pulse summaries
            pulses_data = general_data.get("pulse_info", {}).get("pulses", []) or []
            pulses = []
            for p in pulses_data[:10]:  # Limit to 10 pulses
                pulse = PulseInfo(
                    pulse_id=p.get("id", ""),
                    name=p.get("name", ""),
                    description=p.get("description", "")[:500] if p.get("description") else "",
                    author_name=p.get("author_name", ""),
                    created=p.get("created", ""),
                    modified=p.get("modified", ""),
                    tags=p.get("tags", []) or [],
                    targeted_countries=p.get("targeted_countries", []) or [],
                    malware_families=p.get("malware_families", []) or [],
                    indicator_count=p.get("indicator_count", 0),
                    references=p.get("references", [])[:5] if p.get("references") else [],
                )
                pulses.append(pulse.model_dump())
            
            # Get geo info for IP addresses
            geo_info = None
            if indicator_type in ["IPv4", "IPv6"]:
                geo_info = {
                    "country_code": general_data.get("country_code"),
                    "country_name": general_data.get("country_name"),
                    "city": general_data.get("city"),
                    "region": general_data.get("region"),
                    "latitude": general_data.get("latitude"),
                    "longitude": general_data.get("longitude"),
                    "asn": general_data.get("asn"),
                }
            
            return {
                "status": "success",
                "query_params": query_params,
                "indicator_info": indicator_info.model_dump(),
                "pulses": pulses,
                "pulse_count": len(pulses),
                "geo_info": geo_info,
            }

    except httpx.TimeoutException:
        return {
            "status": "error",
            "error_message": "Request timed out while querying OTX API",
            "query_params": query_params,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error querying OTX: {type(e).__name__}: {str(e)}",
            "query_params": query_params,
        }


async def get_pulse_details(
    pulse_id: str,
) -> dict[str, Any]:
    """
    Get detailed information about a specific OTX pulse.

    Pulses are community-contributed threat intelligence reports that contain
    collections of indicators related to specific threats, campaigns, or malware.

    Args:
        pulse_id: The unique identifier of the pulse to retrieve.

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - pulse: Detailed pulse information including description and metadata
        - indicators: List of indicators of compromise in the pulse
        - indicator_count: Total number of indicators
        - error_message: Error details if the query failed

    Example:
        >>> result = await get_pulse_details("507f1f77bcf86cd799439011")
        >>> print(f"Pulse: {result['pulse']['name']}")
        >>> for ioc in result['indicators']:
        ...     print(f"  {ioc['type']}: {ioc['indicator']}")
    """
    if not settings.has_otx_key:
        return {
            "status": "error",
            "error_message": "OTX_API_KEY not configured. Get your free API key at https://otx.alienvault.com",
            "pulse_id": pulse_id,
        }

    query_params = {
        "pulse_id": pulse_id,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get pulse details
            pulse_url = f"{OTX_API_BASE_URL}/pulses/{pulse_id}"
            response = await client.get(pulse_url, headers=_get_headers())
            
            if response.status_code == 404:
                return {
                    "status": "error",
                    "error_message": f"Pulse {pulse_id} not found",
                    "query_params": query_params,
                }
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error_message": f"OTX API error: HTTP {response.status_code}",
                    "query_params": query_params,
                }
            
            data = response.json()
            
            # Build pulse info
            pulse = PulseInfo(
                pulse_id=data.get("id", pulse_id),
                name=data.get("name", ""),
                description=data.get("description", ""),
                author_name=data.get("author_name", ""),
                created=data.get("created", ""),
                modified=data.get("modified", ""),
                tags=data.get("tags", []) or [],
                targeted_countries=data.get("targeted_countries", []) or [],
                malware_families=data.get("malware_families", []) or [],
                indicator_count=len(data.get("indicators", [])),
                references=data.get("references", []) or [],
            )
            
            # Extract indicators (limit to 50)
            indicators_raw = data.get("indicators", [])[:50]
            indicators = []
            for ind in indicators_raw:
                indicators.append({
                    "indicator": ind.get("indicator", ""),
                    "type": ind.get("type", ""),
                    "title": ind.get("title", ""),
                    "description": ind.get("description", ""),
                    "created": ind.get("created", ""),
                })
            
            return {
                "status": "success",
                "query_params": query_params,
                "pulse": pulse.model_dump(),
                "indicators": indicators,
                "indicator_count": len(indicators_raw),
                "total_indicators": data.get("indicator_count", len(indicators_raw)),
            }

    except httpx.TimeoutException:
        return {
            "status": "error",
            "error_message": "Request timed out while querying OTX API",
            "query_params": query_params,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error querying OTX: {type(e).__name__}: {str(e)}",
            "query_params": query_params,
        }


async def search_pulses(
    query: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for threat intelligence pulses in OTX.

    Search the OTX database for pulses (threat reports) matching keywords.
    Useful for researching specific threats, malware families, or APT groups.

    Args:
        query: Search terms (e.g., "APT28", "ransomware", "Ukraine cyberattack").
        limit: Maximum number of pulses to return (1-50). Default: 20.

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - pulses: List of matching pulses with metadata
        - pulse_count: Number of pulses returned
        - error_message: Error details if the query failed

    Example:
        >>> result = await search_pulses("APT28 Russia")
        >>> for pulse in result['pulses']:
        ...     print(f"{pulse['name']} ({pulse['indicator_count']} indicators)")
    """
    if not settings.has_otx_key:
        return {
            "status": "error",
            "error_message": "OTX_API_KEY not configured. Get your free API key at https://otx.alienvault.com",
            "query": query,
        }

    limit = max(1, min(50, limit))
    
    query_params = {
        "query": query,
        "limit": limit,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search pulses
            search_url = f"{OTX_API_BASE_URL}/search/pulses"
            params = {"q": query, "limit": limit}
            response = await client.get(search_url, headers=_get_headers(), params=params)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error_message": f"OTX API error: HTTP {response.status_code}",
                    "query_params": query_params,
                }
            
            data = response.json()
            results = data.get("results", [])
            
            pulses = []
            for p in results:
                pulse = PulseInfo(
                    pulse_id=p.get("id", ""),
                    name=p.get("name", ""),
                    description=p.get("description", "")[:500] if p.get("description") else "",
                    author_name=p.get("author_name", ""),
                    created=p.get("created", ""),
                    modified=p.get("modified", ""),
                    tags=p.get("tags", []) or [],
                    targeted_countries=p.get("targeted_countries", []) or [],
                    malware_families=p.get("malware_families", []) or [],
                    indicator_count=p.get("indicator_count", 0),
                    references=p.get("references", [])[:5] if p.get("references") else [],
                )
                pulses.append(pulse.model_dump())
            
            return {
                "status": "success",
                "query_params": query_params,
                "pulses": pulses,
                "pulse_count": len(pulses),
            }

    except httpx.TimeoutException:
        return {
            "status": "error",
            "error_message": "Request timed out while querying OTX API",
            "query_params": query_params,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error querying OTX: {type(e).__name__}: {str(e)}",
            "query_params": query_params,
        }

