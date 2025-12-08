"""
NASA FIRMS Integration Module.

This module provides tools to query the NASA Fire Information for Resource Management
System (FIRMS) for thermal anomalies detection, which can indicate fires, explosions,
or other heat-generating events.
"""

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.shared.config import settings


class ThermalAnomaly(BaseModel):
    """Represents a single thermal anomaly detected by satellite."""

    latitude: float = Field(description="Latitude of the anomaly")
    longitude: float = Field(description="Longitude of the anomaly")
    brightness: float = Field(description="Brightness temperature in Kelvin")
    scan: float = Field(description="Scan pixel size")
    track: float = Field(description="Track pixel size")
    acq_date: str = Field(description="Acquisition date")
    acq_time: str = Field(description="Acquisition time (HHMM)")
    satellite: str = Field(description="Satellite source (e.g., VIIRS, MODIS)")
    confidence: str = Field(description="Detection confidence level")
    frp: float | None = Field(default=None, description="Fire Radiative Power in MW")


class FIRMSResponse(BaseModel):
    """Response model for NASA FIRMS API queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    anomaly_count: int = Field(description="Number of anomalies detected")
    anomalies: list[ThermalAnomaly] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="NASA FIRMS", description="Data source identifier")


async def check_nasa_firms(
    latitude: float,
    longitude: float,
    day_range: int = 7,
    radius_km: int = 50,
) -> dict[str, Any]:
    """
    Query NASA FIRMS API to detect thermal anomalies (fires, explosions) near a location.

    This tool queries satellite data from NASA's Fire Information for Resource Management
    System to identify heat signatures that could indicate:
    - Active fires or wildfires
    - Industrial explosions
    - Military strikes or bombardments
    - Large-scale burning events

    Args:
        latitude: Latitude of the center point to search (-90 to 90).
        longitude: Longitude of the center point to search (-180 to 180).
        day_range: Number of days to look back (1-10). Default is 7 days.
        radius_km: Search radius in kilometers from the center point (1-100). Default is 50km.

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'
        - query_params: The parameters used for the query
        - anomaly_count: Number of thermal anomalies found
        - anomalies: List of detected thermal anomalies with coordinates and metadata
        - error_message: Error details if the query failed

    Example:
        >>> result = await check_nasa_firms(latitude=50.4501, longitude=30.5234, day_range=3)
        >>> print(f"Found {result['anomaly_count']} thermal anomalies near Kyiv")
    """
    api_key = settings.nasa_firms_api_key

    if not api_key:
        return FIRMSResponse(
            status="error",
            query_params={
                "latitude": latitude,
                "longitude": longitude,
                "day_range": day_range,
            },
            anomaly_count=0,
            error_message="TOOL DISABLED: NASA_FIRMS_API_KEY is not configured. "
            "This tool cannot return any thermal anomaly data. "
            "To enable: set NASA_FIRMS_API_KEY in your .env file. "
            "Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/area/",
        ).model_dump()

    # Validate inputs
    day_range = max(1, min(10, day_range))
    radius_km = max(1, min(100, radius_km))

    # Calculate bounding box from center point and radius
    # Approximate: 1 degree latitude ≈ 111km
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * abs(max(0.1, abs(latitude)) ** 0.5))

    min_lat = max(-90, latitude - lat_delta)
    max_lat = min(90, latitude + lat_delta)
    min_lon = max(-180, longitude - lon_delta)
    max_lon = min(180, longitude + lon_delta)

    # FIRMS API endpoint for area queries
    # Using VIIRS_SNPP for better resolution
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    url = f"{base_url}/{api_key}/VIIRS_SNPP_NRT/{min_lon},{min_lat},{max_lon},{max_lat}/{day_range}"

    query_params = {
        "latitude": latitude,
        "longitude": longitude,
        "day_range": day_range,
        "radius_km": radius_km,
        "bounding_box": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Parse CSV response
            lines = response.text.strip().split("\n")

            if len(lines) <= 1:
                return FIRMSResponse(
                    status="success",
                    query_params=query_params,
                    anomaly_count=0,
                    anomalies=[],
                ).model_dump()

            # Parse header and data
            headers = lines[0].lower().split(",")
            anomalies: list[ThermalAnomaly] = []

            for line in lines[1:]:
                values = line.split(",")
                if len(values) < len(headers):
                    continue

                row = dict(zip(headers, values))

                try:
                    anomaly = ThermalAnomaly(
                        latitude=float(row.get("latitude", 0)),
                        longitude=float(row.get("longitude", 0)),
                        brightness=float(row.get("bright_ti4", row.get("brightness", 0))),
                        scan=float(row.get("scan", 0)),
                        track=float(row.get("track", 0)),
                        acq_date=row.get("acq_date", ""),
                        acq_time=row.get("acq_time", ""),
                        satellite=row.get("satellite", "VIIRS"),
                        confidence=row.get("confidence", "nominal"),
                        frp=float(row["frp"]) if row.get("frp") else None,
                    )
                    anomalies.append(anomaly)
                except (ValueError, KeyError):
                    # Skip malformed rows
                    continue

            return FIRMSResponse(
                status="success",
                query_params=query_params,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
            ).model_dump()

    except httpx.TimeoutException:
        return FIRMSResponse(
            status="error",
            query_params=query_params,
            anomaly_count=0,
            error_message="Request to NASA FIRMS timed out. Try again later.",
        ).model_dump()

    except httpx.HTTPStatusError as e:
        return FIRMSResponse(
            status="error",
            query_params=query_params,
            anomaly_count=0,
            error_message=f"NASA FIRMS API returned HTTP {e.response.status_code}: {e.response.text[:200]}",
        ).model_dump()

    except Exception as e:
        return FIRMSResponse(
            status="error",
            query_params=query_params,
            anomaly_count=0,
            error_message=f"Unexpected error querying NASA FIRMS: {str(e)}",
        ).model_dump()

