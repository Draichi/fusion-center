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
    # Using the public IODA API from CAIDA
    ioda_base_url = "https://api.ioda.inetintel.cc.gatech.edu/v2"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Query IODA for country-level signals
            # Entity type: country, Entity code: ISO country code
            signals_url = (
                f"{ioda_base_url}/signals/country/{resolved_code}"
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
    """Parse IODA API response into our standard format."""
    outages: list[OutageEvent] = []

    # Extract signals data
    signals = data.get("data", {}).get("signals", [])

    # Determine overall status from recent signals
    recent_score = 1.0  # Default to normal
    if signals:
        # Get the most recent signal values
        latest_signals = signals[-10:] if len(signals) > 10 else signals
        avg_score = sum(s.get("score", 1.0) for s in latest_signals) / len(latest_signals)
        recent_score = avg_score

    # Determine status level
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
        active_probes=None,
        last_updated=datetime.utcnow().isoformat(),
    )

    # Identify outage events from signal drops
    for signal in signals:
        score = signal.get("score", 1.0)
        if score < 0.7:  # Significant drop
            severity = "critical" if score < 0.3 else "high" if score < 0.5 else "medium"
            outages.append(
                OutageEvent(
                    region=country_name,
                    region_code=country_code,
                    start_time=signal.get("time", ""),
                    end_time=None,
                    severity=severity,
                    data_source="IODA BGP/Active Probing",
                    metrics={"visibility_score": score},
                )
            )

    return IODAResponse(
        status="success",
        query_params=query_params,
        current_status=current_status,
        recent_outages=outages,
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

    Note: This endpoint uses public Cloudflare Radar data. For detailed metrics,
    a Cloudflare API token is required.

    Args:
        country_code: ISO 2-letter country code.
        metric: Type of metric to query: 'traffic', 'attacks', or 'routing'.

    Returns:
        Traffic and security metrics for the specified country.
    """
    country_code = country_code.upper()
    country_name = COUNTRY_MAPPING.get(country_code, country_code)

    query_params = {
        "country_code": country_code,
        "metric": metric,
        "query_time": datetime.utcnow().isoformat(),
    }

    # Cloudflare Radar public summary page
    # Note: Full API access requires authentication
    radar_url = f"https://radar.cloudflare.com/api/v1/traffic/summary?location={country_code}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(radar_url)

            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "query_params": query_params,
                    "data_source": "Cloudflare Radar",
                    "country": country_name,
                    "metrics": data.get("result", {}),
                }
            else:
                return {
                    "status": "error",
                    "query_params": query_params,
                    "error_message": f"TOOL ERROR: Cloudflare Radar returned HTTP {response.status_code}. "
                    "No traffic data available from this request.",
                    "note": "Public Cloudflare Radar API may require authentication for detailed data. "
                    "Set CLOUDFLARE_API_TOKEN in .env for full access.",
                }

    except Exception as e:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": f"Error querying Cloudflare Radar: {str(e)}",
        }

