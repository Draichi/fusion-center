"""
GDELT Project Integration Module.

This module provides tools to query the GDELT (Global Database of Events, Language,
and Tone) Project API for news articles and events related to global conflicts,
political developments, and geopolitical situations.
"""

from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field

from src.shared.config import settings


# GDELT uses country names (not always ISO codes) for sourcecountry filter
# This mapping converts common ISO codes to GDELT-accepted country names
ISO_TO_GDELT_COUNTRY = {
    "UA": "Ukraine",
    "RU": "Russia",
    "US": "US",  # GDELT accepts "US"
    "CN": "China",
    "IR": "Iran",
    "IL": "Israel",
    "PS": "Palestine",
    "SY": "Syria",
    "IQ": "Iraq",
    "AF": "Afghanistan",
    "PK": "Pakistan",
    "IN": "India",
    "KP": "NorthKorea",
    "KR": "SouthKorea",
    "TW": "Taiwan",
    "BY": "Belarus",
    "PL": "Poland",
    "DE": "Germany",
    "FR": "France",
    "GB": "UK",
    "UK": "UK",
    "TR": "Turkey",
    "SA": "SaudiArabia",
    "YE": "Yemen",
    "LB": "Lebanon",
    "JO": "Jordan",
    "EG": "Egypt",
    "LY": "Libya",
    "SD": "Sudan",
    "ET": "Ethiopia",
    "SO": "Somalia",
    "VE": "Venezuela",
    "CU": "Cuba",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    "CO": "Colombia",
    "MM": "Myanmar",
    "TH": "Thailand",
    "VN": "Vietnam",
    "PH": "Philippines",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "AU": "Australia",
    "NZ": "NewZealand",
    "JP": "Japan",
    "NG": "Nigeria",
    "ZA": "SouthAfrica",
    "KE": "Kenya",
}


def _resolve_country_code(country_code: str | None) -> str | None:
    """
    Convert ISO country code to GDELT-accepted country name.
    
    GDELT's sourcecountry filter doesn't accept all ISO codes.
    This function maps common codes to the names GDELT expects.
    """
    if not country_code:
        return None
    
    code_upper = country_code.upper()
    
    # Check if we have a mapping
    if code_upper in ISO_TO_GDELT_COUNTRY:
        return ISO_TO_GDELT_COUNTRY[code_upper]
    
    # If not in mapping, try using the code as-is (might work for some)
    return code_upper


class GDELTArticle(BaseModel):
    """Represents a news article from GDELT."""

    url: str = Field(description="URL of the article")
    title: str = Field(description="Article title")
    seendate: str = Field(description="Date when the article was indexed")
    socialimage: str | None = Field(default=None, description="Social media preview image URL")
    domain: str = Field(description="Source domain")
    language: str = Field(description="Article language code")
    sourcecountry: str | None = Field(default=None, description="Source country code")


class GDELTResponse(BaseModel):
    """Response model for GDELT API queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    article_count: int = Field(description="Number of articles found")
    articles: list[GDELTArticle] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="GDELT Project 2.0", description="Data source identifier")


async def query_gdelt_events(
    keywords: str,
    country_code: str | None = None,
    max_records: int = 50,
    timespan: str = "7d",
) -> dict[str, Any]:
    """
    Query the GDELT Project API for news articles and events matching specified criteria.

    GDELT monitors news media from around the world in over 100 languages, tracking
    events, people, organizations, and themes. This tool is useful for:
    - Monitoring breaking news about conflicts or crises
    - Tracking coverage of specific countries or regions
    - Finding news about military activities, protests, or political events
    - Analyzing media coverage of geopolitical situations

    Args:
        keywords: Search keywords or phrases (e.g., "military strike", "protest Kyiv").
                  Supports boolean operators: AND, OR, NOT.
        country_code: Optional ISO 2-letter country code to filter results (e.g., "UA" for Ukraine,
                      "RU" for Russia, "CN" for China). If not provided, searches globally.
        max_records: Maximum number of articles to return (1-250). Default is 50.
        timespan: Time range to search. Format: number + unit (d=days, h=hours, m=minutes).
                  Examples: "7d" (7 days), "24h" (24 hours), "30m" (30 minutes).
                  Default is "7d".

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'
        - query_params: The parameters used for the query
        - article_count: Number of articles found
        - articles: List of articles with URLs, titles, dates, and source information
        - error_message: Error details if the query failed

    Example:
        >>> result = await query_gdelt_events(
        ...     keywords="military drone strike",
        ...     country_code="UA",
        ...     timespan="3d"
        ... )
        >>> for article in result['articles'][:5]:
        ...     print(f"{article['title']} - {article['domain']}")
    """
    base_url = settings.gdelt_api_base_url
    max_records = max(1, min(250, max_records))

    # Build the query
    query_parts = [keywords]
    
    # Resolve country code to GDELT-accepted country name
    gdelt_country = _resolve_country_code(country_code)
    if gdelt_country:
        query_parts.append(f"sourcecountry:{gdelt_country}")

    query = " ".join(query_parts)
    encoded_query = quote_plus(query)

    # GDELT DOC API endpoint
    url = (
        f"{base_url}/doc/doc"
        f"?query={encoded_query}"
        f"&mode=artlist"
        f"&maxrecords={max_records}"
        f"&timespan={timespan}"
        f"&format=json"
        f"&sort=datedesc"
    )

    query_params = {
        "keywords": keywords,
        "country_code": country_code,
        "max_records": max_records,
        "timespan": timespan,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check if response is empty
            response_text = response.text.strip()
            if not response_text:
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message="GDELT API returned empty response. The query may be too restrictive or the service is temporarily unavailable.",
                ).model_dump()

            # Check if response looks like HTML (error page)
            if response_text.startswith("<!") or response_text.startswith("<html"):
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message="GDELT API returned HTML instead of JSON. The service may be experiencing issues.",
                ).model_dump()

            # Try to parse JSON
            try:
                data = response.json()
            except Exception as json_err:
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message=f"Failed to parse GDELT response as JSON: {str(json_err)}. Response preview: {response_text[:100]}",
                ).model_dump()

            # GDELT returns articles in an "articles" array
            raw_articles = data.get("articles", [])
            articles: list[GDELTArticle] = []

            for item in raw_articles:
                try:
                    article = GDELTArticle(
                        url=item.get("url", ""),
                        title=item.get("title", "No title"),
                        seendate=item.get("seendate", ""),
                        socialimage=item.get("socialimage"),
                        domain=item.get("domain", "unknown"),
                        language=item.get("language", "en"),
                        sourcecountry=item.get("sourcecountry"),
                    )
                    articles.append(article)
                except Exception:
                    # Skip malformed articles
                    continue

            return GDELTResponse(
                status="success",
                query_params=query_params,
                article_count=len(articles),
                articles=articles,
            ).model_dump()

    except httpx.TimeoutException:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message="Request to GDELT API timed out. Try again later or narrow your search.",
        ).model_dump()

    except httpx.HTTPStatusError as e:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message=f"GDELT API returned HTTP {e.response.status_code}: {e.response.text[:200]}",
        ).model_dump()

    except httpx.ConnectError as e:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message=f"Failed to connect to GDELT API: {str(e)}. Check network connectivity.",
        ).model_dump()

    except Exception as e:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message=f"Unexpected error querying GDELT: {type(e).__name__}: {str(e)}",
        ).model_dump()


async def query_gdelt_geo(
    keywords: str,
    latitude: float,
    longitude: float,
    radius_km: int = 100,
    max_records: int = 50,
) -> dict[str, Any]:
    """
    Query GDELT for news articles near a specific geographic location.

    This tool allows searching for news coverage of events happening near a specific
    coordinate point, useful for correlating satellite data with news reports.

    Args:
        keywords: Search keywords or phrases.
        latitude: Latitude of the center point (-90 to 90).
        longitude: Longitude of the center point (-180 to 180).
        radius_km: Search radius in kilometers (1-500). Default is 100km.
        max_records: Maximum number of articles to return (1-250). Default is 50.

    Returns:
        A dictionary containing matching articles near the specified location.

    Example:
        >>> result = await query_gdelt_geo(
        ...     keywords="explosion",
        ...     latitude=50.4501,
        ...     longitude=30.5234,
        ...     radius_km=50
        ... )
    """
    base_url = settings.gdelt_api_base_url
    max_records = max(1, min(250, max_records))
    radius_km = max(1, min(500, radius_km))

    # Build geo-constrained query
    geo_query = f"near:{latitude},{longitude},{radius_km}km"
    full_query = f"{keywords} {geo_query}"
    encoded_query = quote_plus(full_query)

    url = (
        f"{base_url}/doc/doc"
        f"?query={encoded_query}"
        f"&mode=artlist"
        f"&maxrecords={max_records}"
        f"&format=json"
        f"&sort=datedesc"
    )

    query_params = {
        "keywords": keywords,
        "latitude": latitude,
        "longitude": longitude,
        "radius_km": radius_km,
        "max_records": max_records,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check if response is empty
            response_text = response.text.strip()
            if not response_text:
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message="GDELT geo API returned empty response. The location may have no coverage or the query is too restrictive.",
                ).model_dump()

            # Check if response looks like HTML (error page)
            if response_text.startswith("<!") or response_text.startswith("<html"):
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message="GDELT geo API returned HTML instead of JSON. The service may be experiencing issues.",
                ).model_dump()

            # Try to parse JSON
            try:
                data = response.json()
            except Exception as json_err:
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message=f"Failed to parse GDELT geo response as JSON: {str(json_err)}. Response preview: {response_text[:100]}",
                ).model_dump()

            raw_articles = data.get("articles", [])
            articles: list[GDELTArticle] = []

            for item in raw_articles:
                try:
                    article = GDELTArticle(
                        url=item.get("url", ""),
                        title=item.get("title", "No title"),
                        seendate=item.get("seendate", ""),
                        socialimage=item.get("socialimage"),
                        domain=item.get("domain", "unknown"),
                        language=item.get("language", "en"),
                        sourcecountry=item.get("sourcecountry"),
                    )
                    articles.append(article)
                except Exception:
                    continue

            return GDELTResponse(
                status="success",
                query_params=query_params,
                article_count=len(articles),
                articles=articles,
            ).model_dump()

    except httpx.TimeoutException:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message="Request to GDELT geo API timed out. Try again later or use a smaller radius.",
        ).model_dump()

    except httpx.HTTPStatusError as e:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message=f"GDELT geo API returned HTTP {e.response.status_code}: {e.response.text[:200]}",
        ).model_dump()

    except httpx.ConnectError as e:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message=f"Failed to connect to GDELT geo API: {str(e)}. Check network connectivity.",
        ).model_dump()

    except Exception as e:
        return GDELTResponse(
            status="error",
            query_params=query_params,
            article_count=0,
            error_message=f"Error querying GDELT geo search: {type(e).__name__}: {str(e)}",
        ).model_dump()

