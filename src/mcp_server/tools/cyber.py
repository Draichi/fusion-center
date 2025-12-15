"""
Internet Infrastructure Monitoring Module.

This module provides tools to query internet outage and connectivity data
from IODA (Internet Outage Detection and Analysis) and other public sources.
"""

from datetime import datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel, Field


class OutageEvent(BaseModel):
    """Represents an internet outage or connectivity anomaly."""

    region: str = Field(description="Affected region or country")
    region_code: str = Field(description="Region/country code")
    start_time: str | None = Field(default=None, description="Outage start time (ISO format)")
    end_time: str | None = Field(default=None, description="Outage end time if resolved")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    data_source: str = Field(description="Source of the outage data")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Additional metrics about the outage"
    )


class ConnectivityStatus(BaseModel):
    """Current connectivity status for a region."""

    region: str = Field(description="Region name")
    region_code: str = Field(description="Region/country code")
    status: str = Field(description="Current status: normal, degraded, outage")
    bgp_visibility: float | None = Field(
        default=None, description="BGP visibility percentage (0-100)"
    )
    active_probes: int | None = Field(default=None, description="Number of active measurement probes")
    last_updated: str = Field(description="Last update timestamp")


class IODAResponse(BaseModel):
    """Response model for IODA/connectivity queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    current_status: ConnectivityStatus | None = Field(default=None)
    recent_outages: list[OutageEvent] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="IODA/Cloudflare Radar", description="Data source identifier")


# Country code to name mapping for common targets
COUNTRY_MAPPING = {
    "UA": "Ukraine",
    "RU": "Russia",
    "CN": "China",
    "IR": "Iran",
    "KP": "North Korea",
    "SY": "Syria",
    "IQ": "Iraq",
    "AF": "Afghanistan",
    "YE": "Yemen",
    "LB": "Lebanon",
    "PS": "Palestine",
    "IL": "Israel",
    "TW": "Taiwan",
    "MM": "Myanmar",
    "ET": "Ethiopia",
    "SD": "Sudan",
    "VE": "Venezuela",
    "CU": "Cuba",
    "BY": "Belarus",
    "US": "United States",
    "GB": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
}


async def check_internet_outages(
    country_code: str | None = None,
    region_name: str | None = None,
    hours_back: int = 24,
) -> dict[str, Any]:
    """
    Check for recent internet outages and connectivity issues in a region.

    This tool queries public internet measurement data to detect connectivity
    disruptions that could indicate:
    - Government-imposed internet shutdowns
    - Infrastructure damage from conflicts
    - Cyber attacks on network infrastructure
    - Natural disasters affecting connectivity
    - Cable cuts or routing anomalies

    Args:
        country_code: ISO 2-letter country code (e.g., "UA", "RU", "IR").
                      Takes precedence over region_name if both provided.
        region_name: Alternative to country_code - full region name (e.g., "Ukraine").
        hours_back: Number of hours to look back for outages (1-168). Default is 24 hours.

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'
        - query_params: The parameters used for the query
        - current_status: Current connectivity status for the region
        - recent_outages: List of recent outage events
        - error_message: Error details if the query failed

    Example:
        >>> result = await check_internet_outages(country_code="UA", hours_back=48)
        >>> print(f"Ukraine connectivity: {result['current_status']['status']}")
        >>> for outage in result['recent_outages']:
        ...     print(f"Outage detected: {outage['severity']} - {outage['start_time']}")
    """
    hours_back = max(1, min(168, hours_back))

    # Resolve region
    resolved_code = None
    resolved_name = None

    if country_code:
        resolved_code = country_code.upper()
        resolved_name = COUNTRY_MAPPING.get(resolved_code, country_code)
    elif region_name:
        # Try to find country code from name
        resolved_name = region_name
        for code, name in COUNTRY_MAPPING.items():
            if name.lower() == region_name.lower():
                resolved_code = code
                break
        if not resolved_code:
            resolved_code = region_name[:2].upper()

    if not resolved_code and not resolved_name:
        return IODAResponse(
            status="error",
            query_params={},
            error_message="Either country_code or region_name must be provided.",
        ).model_dump()

    query_params = {
        "country_code": resolved_code,
        "region_name": resolved_name,
        "hours_back": hours_back,
        "query_time": datetime.utcnow().isoformat(),
    }

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours_back)

    # IODA API endpoints
    # Using the public IODA API from Georgia Tech (formerly CAIDA)
    # Docs: https://api.ioda.inetintel.cc.gatech.edu/v2/
    ioda_base_url = "https://api.ioda.inetintel.cc.gatech.edu/v2"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Query IODA for country-level signals
            # Endpoint: /signals/raw/{entityType}/{entityCode}
            # Entity type: country, region, asn
            # Entity code: ISO 2-letter country code (e.g., UA, US, RU)
            signals_url = (
                f"{ioda_base_url}/signals/raw/country/{resolved_code}"
                f"?from={int(start_time.timestamp())}"
                f"&until={int(end_time.timestamp())}"
            )

            response = await client.get(signals_url)

            if response.status_code == 200:
                data = response.json()
                return _parse_ioda_response(data, query_params, resolved_code, resolved_name)
            elif response.status_code == 404:
                # Region not found in IODA, return simulated status
                return _get_fallback_status(query_params, resolved_code, resolved_name)
            else:
                response.raise_for_status()

    except httpx.TimeoutException:
        return IODAResponse(
            status="error",
            query_params=query_params,
            error_message="Request to IODA API timed out. Try again later.",
        ).model_dump()

    except httpx.HTTPStatusError as e:
        # Fall back to basic status if IODA fails
        return _get_fallback_status(
            query_params,
            resolved_code,
            resolved_name,
            error_note=f"IODA API returned {e.response.status_code}",
        )

    except Exception as e:
        return IODAResponse(
            status="error",
            query_params=query_params,
            error_message=f"Unexpected error checking connectivity: {str(e)}",
        ).model_dump()


def _parse_ioda_response(
    data: dict[str, Any],
    query_params: dict[str, Any],
    country_code: str,
    country_name: str,
) -> dict[str, Any]:
    """Parse IODA API response into our standard format.
    
    IODA API v2 response structure:
    - data: list containing one item (the country data)
    - data[0]: list of datasources (bgp, gtr, ping-slash24, etc.)
    - Each datasource has: entityType, entityCode, datasource, values[], step, etc.
    
    Note: Some datasources (gtr-sarima, ping-slash24-loss, ping-slash24-latency)
    have list values instead of scalar values - we skip these for analysis.
    """
    outages: list[OutageEvent] = []
    
    # Datasources with scalar values (usable for connectivity analysis)
    SCALAR_DATASOURCES = {"bgp", "gtr", "gtr-norm", "merit-nt", "ping-slash24"}
    
    # Extract datasources from the nested structure
    # data is a list, data[0] is a list of datasource objects
    raw_data = data.get("data", [])
    datasources = raw_data[0] if raw_data and isinstance(raw_data[0], list) else []
    
    # Find BGP datasource for visibility metrics (most reliable for connectivity)
    bgp_data = None
    ping_data = None
    datasource_info = {}
    
    for ds in datasources:
        ds_name = ds.get("datasource", "")
        datasource_info[ds_name] = {
            "values_count": len(ds.get("values", [])),
            "step": ds.get("step"),
            "from": ds.get("from"),
            "until": ds.get("until"),
        }
        # Only use datasources with scalar values
        if ds_name not in SCALAR_DATASOURCES:
            continue
        if ds_name == "bgp":
            bgp_data = ds
        elif ds_name == "ping-slash24":
            ping_data = ds
    
    # Calculate connectivity score from BGP or ping data
    recent_score = 1.0  # Default to normal
    active_probes = None
    
    # Prefer BGP data for visibility score
    primary_data = bgp_data or ping_data
    if primary_data:
        values = primary_data.get("values", [])
        if values:
            # Get recent values (last 10 data points), filter out non-numeric values
            recent_values = [
                v for v in values[-10:] 
                if v is not None and isinstance(v, (int, float))
            ]
            if recent_values:
                # Normalize: compare to max value to get a ratio
                max_val = max(recent_values) if recent_values else 1
                avg_val = sum(recent_values) / len(recent_values)
                
                # If there's significant variation, there might be issues
                if max_val > 0:
                    recent_score = avg_val / max_val
                
                # Store active probes count if using ping data
                if primary_data == ping_data:
                    active_probes = int(avg_val) if avg_val else None

    # Determine status level based on score
    if recent_score >= 0.8:
        status_level = "normal"
    elif recent_score >= 0.5:
        status_level = "degraded"
    else:
        status_level = "outage"

    current_status = ConnectivityStatus(
        region=country_name,
        region_code=country_code,
        status=status_level,
        bgp_visibility=recent_score * 100,
        active_probes=active_probes,
        last_updated=datetime.utcnow().isoformat(),
    )

    # Check for significant drops in scalar datasources to identify outages
    for ds in datasources:
        ds_name = ds.get("datasource", "unknown")
        
        # Skip datasources with non-scalar values
        if ds_name not in SCALAR_DATASOURCES:
            continue
            
        values = ds.get("values", [])
        step = ds.get("step", 300)  # Default 5 min intervals
        start_time_unix = ds.get("from", 0)
        
        if not values or len(values) < 2:
            continue
        
        # Filter to only numeric values
        numeric_values = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float)) and v is not None]
        if len(numeric_values) < 2:
            continue
            
        # Calculate baseline (average of first half)
        half_idx = len(numeric_values) // 2
        baseline_values = [v for _, v in numeric_values[:half_idx]]
        if not baseline_values:
            continue
        baseline = sum(baseline_values) / len(baseline_values)
        
        if baseline == 0:
            continue
            
        # Look for significant drops (< 70% of baseline)
        for i, val in numeric_values:
            ratio = val / baseline
            if ratio < 0.7:  # Significant drop
                severity = "critical" if ratio < 0.3 else "high" if ratio < 0.5 else "medium"
                event_time = datetime.utcfromtimestamp(start_time_unix + i * step)
                outages.append(
                    OutageEvent(
                        region=country_name,
                        region_code=country_code,
                        start_time=event_time.isoformat(),
                        end_time=None,
                        severity=severity,
                        data_source=f"IODA {ds_name}",
                        metrics={"value": val, "baseline": baseline, "ratio": round(ratio, 3)},
                    )
                )

    return IODAResponse(
        status="success",
        query_params=query_params,
        current_status=current_status,
        recent_outages=outages,
        data_source=f"IODA (datasources: {', '.join(datasource_info.keys())})",
    ).model_dump()


def _get_fallback_status(
    query_params: dict[str, Any],
    country_code: str,
    country_name: str,
    error_note: str | None = None,
) -> dict[str, Any]:
    """Return a fallback status when IODA data is unavailable."""
    # Check if the code looks invalid (not a real ISO country code)
    is_invalid_code = len(country_code) != 2 or country_code not in COUNTRY_MAPPING
    
    if is_invalid_code:
        error_msg = (
            f"IODA only supports country-level data (e.g., 'UA' for Ukraine). "
            f"'{country_name}' appears to be a region/city, not a country. "
            f"Try using the country code instead (e.g., country_code='UA' for Ukraine)."
        )
    else:
        error_msg = error_note or f"IODA returned no data for {country_name} ({country_code}). The API may be temporarily unavailable."

    current_status = ConnectivityStatus(
        region=country_name,
        region_code=country_code,
        status="no_data",
        bgp_visibility=None,
        active_probes=None,
        last_updated=datetime.utcnow().isoformat(),
    )

    return IODAResponse(
        status="error" if is_invalid_code else "success",
        query_params=query_params,
        current_status=current_status,
        recent_outages=[],
        error_message=error_msg,
        data_source="IODA (no data available)",
    ).model_dump()


async def check_cloudflare_radar(
    country_code: str,
    metric: str = "traffic",
) -> dict[str, Any]:
    """
    Query Cloudflare Radar for traffic and attack data.

    Requires CLOUDFLARE_API_TOKEN with Account > Radar > Read permissions.
    Get your token at: https://dash.cloudflare.com/profile/api-tokens

    Args:
        country_code: ISO 2-letter country code.
        metric: Type of metric to query: 'traffic', 'attacks', or 'routing'.

    Returns:
        Traffic and security metrics for the specified country.
    """
    from src.shared.config import settings
    
    country_code = country_code.upper()
    country_name = COUNTRY_MAPPING.get(country_code, country_code)

    query_params = {
        "country_code": country_code,
        "metric": metric,
        "query_time": datetime.utcnow().isoformat(),
    }

    # Check if API token is configured
    api_token = settings.cloudflare_api_token
    if not api_token:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": "TOOL DISABLED: CLOUDFLARE_API_TOKEN not configured. "
            "Create a Custom Token with Account > Radar > Read permissions at: "
            "https://dash.cloudflare.com/profile/api-tokens",
        }

    # Cloudflare Radar API v4 - correct endpoint
    # Docs: https://developers.cloudflare.com/radar/get-started/first-request/
    base_url = "https://api.cloudflare.com/client/v4/radar"
    
    # Select endpoint based on metric type
    if metric == "traffic":
        # HTTP traffic summary by device type for a specific location
        endpoint = f"{base_url}/http/summary/device_type"
        params = {"location": country_code, "dateRange": "7d", "format": "json"}
    elif metric == "attacks":
        # Layer 3/4 attack summary
        endpoint = f"{base_url}/attacks/layer3/summary"
        params = {"location": country_code, "dateRange": "7d", "format": "json"}
    elif metric == "routing":
        # BGP routing stats
        endpoint = f"{base_url}/bgp/routes/stats"
        params = {"location": country_code, "format": "json"}
    else:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": f"Unknown metric type: '{metric}'. Use 'traffic', 'attacks', or 'routing'.",
        }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                
                if not data.get("success", False):
                    errors = data.get("errors", [])
                    error_msg = errors[0].get("message", "Unknown error") if errors else "API returned success=false"
                    return {
                        "status": "error",
                        "query_params": query_params,
                        "error_message": f"Cloudflare API error: {error_msg}",
                        "raw_response": data,
                    }
                
                result = data.get("result", {})
                meta = result.get("meta", {})
                
                return {
                    "status": "success",
                    "query_params": query_params,
                    "data_source": "Cloudflare Radar API v4",
                    "country": country_name,
                    "metric_type": metric,
                    "metrics": result,
                    "date_range": meta.get("dateRange", {}),
                    "confidence": meta.get("confidenceInfo", {}).get("level", "unknown"),
                }
            elif response.status_code == 401:
                return {
                    "status": "error",
                    "query_params": query_params,
                    "error_message": "Authentication failed. Check your CLOUDFLARE_API_TOKEN. "
                    "Ensure the token has Account > Radar > Read permissions.",
                }
            elif response.status_code == 403:
                return {
                    "status": "error",
                    "query_params": query_params,
                    "error_message": "Access forbidden. Your API token may not have Radar permissions. "
                    "Create a new token with Account > Radar > Read at: "
                    "https://dash.cloudflare.com/profile/api-tokens",
                }
            elif response.status_code == 400:
                # Try to parse error details
                try:
                    error_data = response.json()
                    errors = error_data.get("errors", [])
                    if errors:
                        error_msg = errors[0].get("message", "Bad request")
                        error_code = errors[0].get("code", "")
                        return {
                            "status": "error",
                            "query_params": query_params,
                            "error_message": f"Cloudflare Radar API error (code {error_code}): {error_msg}. "
                            f"The endpoint '{endpoint}' may not support the 'location' parameter, "
                            f"or the API structure may have changed. Try using metric='traffic' instead.",
                        }
                except:
                    pass
                
                return {
                    "status": "error",
                    "query_params": query_params,
                    "error_message": f"Cloudflare Radar returned HTTP 400 (Bad Request). "
                    f"The endpoint may not support the requested parameters. "
                    f"Try using metric='traffic' instead of '{metric}'.",
                    "raw_response": response.text[:500],
                }
            else:
                return {
                    "status": "error",
                    "query_params": query_params,
                    "error_message": f"Cloudflare Radar returned HTTP {response.status_code}: {response.text[:200]}",
                }

    except httpx.TimeoutException:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": "Request to Cloudflare Radar timed out.",
        }
    except Exception as e:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": f"Error querying Cloudflare Radar: {type(e).__name__}: {str(e)}",
        }


async def get_ioda_outages(
    entity_type: str = "country",
    entity_code: str | None = None,
    days_back: int = 7,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Query IODA for detected internet outage events.

    This tool queries the IODA (Internet Outage Detection and Analysis) API
    for detected outage events. Unlike check_internet_outages which shows
    current connectivity status, this returns a list of actual outage events
    detected by IODA's monitoring systems.

    IODA detects outages using multiple data sources:
    - BGP: Border Gateway Protocol route visibility
    - Active Probing (ping-slash24): Active measurement probes
    - Network Telescope (merit-nt): Darknet traffic analysis
    - Google Transparency Report (gtr): Google product accessibility

    Args:
        entity_type: Type of entity to query. Options:
                     - 'country': Country-level outages (use ISO 2-letter code)
                     - 'region': Sub-country region outages
                     - 'asn': Autonomous System Number outages
                     Leave empty for global outages.
        entity_code: Code for the entity (e.g., 'UA' for Ukraine, 'AS12345' for ASN).
                     Leave empty for all entities of the specified type.
        days_back: Number of days to look back (1-90). Default: 7 days.
        limit: Maximum number of outage events to return (1-100). Default: 50.

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - query_params: Parameters used for the query
        - outage_count: Number of outage events found
        - outages: List of detected outage events with details
        - error_message: Error details if the query failed

    Example:
        >>> result = await get_ioda_outages(entity_type="country", entity_code="UA", days_back=7)
        >>> for outage in result['outages']:
        ...     print(f"{outage['location_name']}: score={outage['score']}")
    """
    # Validate parameters
    days_back = max(1, min(90, days_back))
    limit = max(1, min(100, limit))

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)

    query_params = {
        "entity_type": entity_type,
        "entity_code": entity_code,
        "days_back": days_back,
        "limit": limit,
        "from_time": start_time.isoformat(),
        "until_time": end_time.isoformat(),
        "query_time": datetime.utcnow().isoformat(),
    }

    # IODA API endpoint for outage events
    ioda_base_url = "https://api.ioda.inetintel.cc.gatech.edu/v2"
    
    # Build query parameters
    params = {
        "from": int(start_time.timestamp()),
        "until": int(end_time.timestamp()),
        "limit": limit,
    }
    
    # Add entity filters if specified
    if entity_type and entity_code:
        params["entityType"] = entity_type
        params["entityCode"] = entity_code.upper() if entity_type == "country" else entity_code
    elif entity_type:
        params["entityType"] = entity_type

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Query outage events endpoint
            response = await client.get(
                f"{ioda_base_url}/outages/events",
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                
                if data.get("error"):
                    return {
                        "status": "error",
                        "query_params": query_params,
                        "outage_count": 0,
                        "outages": [],
                        "error_message": f"IODA API error: {data.get('error')}",
                    }
                
                raw_outages = data.get("data", [])
                
                # Process and format outage events
                outages = []
                for event in raw_outages:
                    # Parse location info
                    location = event.get("location", "")
                    location_parts = location.split("/") if location else ["unknown", "unknown"]
                    loc_type = location_parts[0] if len(location_parts) > 0 else "unknown"
                    loc_code = location_parts[1] if len(location_parts) > 1 else "unknown"
                    
                    # Calculate end time from start + duration
                    start_ts = event.get("start", 0)
                    duration = event.get("duration", 0)
                    end_ts = start_ts + duration if start_ts and duration else None
                    
                    # Determine severity from score
                    score = event.get("score", 0)
                    if score >= 10000:
                        severity = "critical"
                    elif score >= 1000:
                        severity = "high"
                    elif score >= 100:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    outages.append({
                        "location": location,
                        "location_name": event.get("location_name", loc_code),
                        "location_type": loc_type,
                        "location_code": loc_code,
                        "datasource": event.get("datasource", "unknown"),
                        "method": event.get("method", "unknown"),
                        "score": score,
                        "severity": severity,
                        "start_time": datetime.utcfromtimestamp(start_ts).isoformat() if start_ts else None,
                        "end_time": datetime.utcfromtimestamp(end_ts).isoformat() if end_ts else None,
                        "duration_seconds": duration,
                        "duration_human": _format_duration(duration) if duration else None,
                        "status": "ongoing" if event.get("status") == 0 else "resolved",
                    })
                
                # Sort by score (most severe first)
                outages.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                return {
                    "status": "success",
                    "query_params": query_params,
                    "outage_count": len(outages),
                    "outages": outages,
                    "data_source": "IODA Outage Detection",
                }
            
            elif response.status_code == 404:
                return {
                    "status": "success",
                    "query_params": query_params,
                    "outage_count": 0,
                    "outages": [],
                    "note": "No outages found for the specified criteria.",
                    "data_source": "IODA Outage Detection",
                }
            else:
                return {
                    "status": "error",
                    "query_params": query_params,
                    "outage_count": 0,
                    "outages": [],
                    "error_message": f"IODA API returned HTTP {response.status_code}",
                }

    except httpx.TimeoutException:
        return {
            "status": "error",
            "query_params": query_params,
            "outage_count": 0,
            "outages": [],
            "error_message": "Request to IODA API timed out. Try again later.",
        }

    except Exception as e:
        return {
            "status": "error",
            "query_params": query_params,
            "outage_count": 0,
            "outages": [],
            "error_message": f"Unexpected error querying IODA: {type(e).__name__}: {str(e)}",
        }


def _format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h" if hours else f"{days}d"
