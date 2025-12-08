"""
GDELT Project Integration Module.

This module provides tools to query the GDELT (Global Database of Events, Language,
and Tone) Project API for news articles and events related to global conflicts,
political developments, and geopolitical situations.
"""

import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field

from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


def _sanitize_gdelt_query(keywords: str) -> str:
    """
    Sanitize and fix GDELT query syntax to prevent common errors.
    
    GDELT has specific requirements for boolean queries:
    - OR operators must be inside parentheses
    - NOT operators should use proper syntax
    - Multiple terms with OR need to be grouped
    
    This function automatically fixes common LLM-generated query mistakes.
    
    Args:
        keywords: The raw query string
        
    Returns:
        A sanitized query string that conforms to GDELT syntax
        
    Examples:
        >>> _sanitize_gdelt_query("Odessa OR Odesa AND missile")
        "(Odessa OR Odesa) AND missile"
        
        >>> _sanitize_gdelt_query("attack OR strike OR explosion")
        "(attack OR strike OR explosion)"
    """
    if not keywords or not keywords.strip():
        return keywords
    
    original = keywords
    query = keywords.strip()
    
    # Remove any extra whitespace
    query = re.sub(r'\s+', ' ', query)
    
    def has_unparenthesized_or(q: str) -> bool:
        """Check if there's an OR at the top level (not inside parens)."""
        depth = 0
        i = 0
        while i < len(q):
            if q[i] == '(':
                depth += 1
            elif q[i] == ')':
                depth -= 1
            elif depth == 0 and q[i:i+3] == ' OR':
                return True
            i += 1
        return False
    
    def fix_or_grouping(q: str) -> str:
        """Fix OR statements that aren't properly parenthesized."""
        if q.startswith('(') and q.endswith(')'):
            depth = 0
            for i, char in enumerate(q):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                if depth == 0 and i < len(q) - 1:
                    break
            else:
                inner = q[1:-1]
                if not has_unparenthesized_or(inner):
                    return q
        
        and_parts = re.split(r'\s+(AND)\s+', q)
        
        fixed_parts = []
        for part in and_parts:
            if part == 'AND':
                fixed_parts.append(part)
                continue
                
            part = part.strip()
            
            if part.startswith('(') and part.endswith(')'):
                depth = 0
                balanced = True
                for i, char in enumerate(part):
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                    if depth == 0 and i < len(part) - 1:
                        balanced = False
                        break
                if balanced:
                    fixed_parts.append(part)
                    continue
            
            if ' OR ' in part:
                not_match = re.match(r'^(.*?)\s+NOT\s+(.+)$', part)
                if not_match:
                    or_part = not_match.group(1)
                    not_part = not_match.group(2)
                    if ' OR ' in or_part and not (or_part.startswith('(') and or_part.endswith(')')):
                        or_part = f'({or_part})'
                    if ' OR ' in not_part and not (not_part.startswith('(') and not_part.endswith(')')):
                        not_part = f'({not_part})'
                    part = f'{or_part} NOT {not_part}'
                else:
                    if not (part.startswith('(') and part.endswith(')')):
                        part = f'({part})'
            
            fixed_parts.append(part)
        
        return ' '.join(fixed_parts)
    
    if has_unparenthesized_or(query):
        query = fix_or_grouping(query)
    
    query = re.sub(r'\s+', ' ', query).strip()
    
    if query != original:
        logger.debug(f"Sanitized GDELT query: '{original}' -> '{query}'")
    
    return query


class GDELTArticle(BaseModel):
    """Represents a news article from GDELT."""

    url: str = Field(description="URL of the article")
    title: str = Field(description="Article title")
    seendate: str = Field(description="Date when the article was indexed")
    socialimage: str | None = Field(default=None, description="Social media preview image URL")
    domain: str = Field(description="Source domain")
    language: str = Field(description="Article language code")
    sourcecountry: str | None = Field(default=None, description="Source country")


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
    source_country: str | None = None,
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
                  Supports boolean operators: AND, OR (must be in parentheses), NOT.
                  Example: "(Odessa OR Odesa) AND (missile OR drone)"
        source_country: Optional country name to filter by news source origin.
                        Use GDELT country names: Ukraine, Russia, China, Iran, Israel,
                        US, UK, Germany, France, etc. Multi-word names use no spaces
                        (e.g., SouthKorea, NorthKorea, SaudiArabia, SouthAfrica).
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
        ...     keywords="(military OR drone) AND strike",
        ...     source_country="Ukraine",
        ...     timespan="3d"
        ... )
    """
    base_url = settings.gdelt_api_base_url
    max_records = max(1, min(250, max_records))

    # Sanitize keywords to fix common query syntax errors
    sanitized_keywords = _sanitize_gdelt_query(keywords)
    
    # Build the query
    query_parts = [sanitized_keywords]
    
    # Add source country filter if provided
    if source_country:
        # Remove any spaces and use as-is (LLM should provide correct format)
        country = source_country.replace(" ", "")
        query_parts.append(f"sourcecountry:{country}")

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
        "sanitized_keywords": sanitized_keywords,
        "source_country": source_country,
        "max_records": max_records,
        "timespan": timespan,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            response_text = response.text.strip()
            if not response_text:
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message="GDELT API returned empty response. The query may be too restrictive or the service is temporarily unavailable.",
                ).model_dump()

            if response_text.startswith("<!") or response_text.startswith("<html"):
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message="GDELT API returned HTML instead of JSON. The service may be experiencing issues.",
                ).model_dump()

            try:
                data = response.json()
            except Exception as json_err:
                return GDELTResponse(
                    status="error",
                    query_params=query_params,
                    article_count=0,
                    error_message=f"Failed to parse GDELT response as JSON: {str(json_err)}. Response preview: {response_text[:100]}",
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
