"""
Internet Search Tools Module.

This module provides tools for general internet search and specialized leak database search:
1. DuckDuckGo - General web search
2. DDoS Secrets - Leaked/hacked data archive search
"""

from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel, Field

from src.shared.logger import get_logger

logger = get_logger()


# =============================================================================
# DuckDuckGo Search Models
# =============================================================================


class SearchResult(BaseModel):
    """Represents a single search result."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the result")
    snippet: str = Field(description="Brief excerpt/description")
    source: str | None = Field(default=None, description="Source domain/site")


class SearchResponse(BaseModel):
    """Response model for internet search queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    result_count: int = Field(description="Number of results found")
    results: list[SearchResult] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="DuckDuckGo", description="Search engine used")


# =============================================================================
# DDoS Secrets Models
# =============================================================================


class LeakResult(BaseModel):
    """Represents a leaked dataset entry."""

    title: str = Field(description="Title/name of the leak")
    url: str = Field(description="URL to the leak details page")
    description: str | None = Field(default=None, description="Brief description")
    category: str | None = Field(default=None, description="Category/type of leak")
    date: str | None = Field(default=None, description="Publication/discovery date")


class LeakSearchResponse(BaseModel):
    """Response model for DDoS Secrets search queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    result_count: int = Field(description="Number of leaks found")
    results: list[LeakResult] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(
        default="DDoS Secrets (Distributed Denial of Secrets)",
        description="Data source identifier",
    )


# =============================================================================
# DuckDuckGo Search Implementation
# =============================================================================


async def search_web(
    query: str,
    max_results: int = 10,
    region: str | None = None,
    time_range: str = "all",
) -> dict[str, Any]:
    """
    Search the internet using DuckDuckGo.

    DuckDuckGo provides privacy-focused web search without tracking. Use this for:
    - General research and fact-checking
    - Finding information not in specialized databases
    - Discovering news, reports, and articles
    - Background research on organizations, people, or events

    Args:
        query: Search query string (e.g., "Ukraine conflict analysis")
        max_results: Maximum number of results to return (1-50). Default is 10.
        region: Optional region code for localized results (e.g., "us-en", "uk-en",
                "br-pt"). Default is worldwide.
        time_range: Time range filter. Options:
                    - "all": All time (default)
                    - "d": Past day
                    - "w": Past week
                    - "m": Past month
                    - "y": Past year

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'
        - query_params: The parameters used for the query
        - result_count: Number of results found
        - results: List of search results with titles, URLs, and snippets
        - error_message: Error details if the query failed

    Example:
        >>> result = await search_web(
        ...     query="cyber attack Iran infrastructure",
        ...     max_results=10,
        ...     time_range="w"
        ... )
    """
    max_results = max(1, min(50, max_results))

    # Validate time_range
    valid_time_ranges = {"all", "d", "w", "m", "y"}
    if time_range not in valid_time_ranges:
        time_range = "all"

    query_params = {
        "query": query,
        "max_results": max_results,
        "region": region,
        "time_range": time_range,
        "query_time": datetime.utcnow().isoformat(),
    }

    try:
        # Initialize DuckDuckGo search client
        ddgs = DDGS()

        # Perform search
        search_args = {"max_results": max_results}
        if region:
            search_args["region"] = region
        if time_range != "all":
            search_args["timelimit"] = time_range

        raw_results = list(ddgs.text(query, **search_args))

        # Parse results
        results: list[SearchResult] = []
        for item in raw_results:
            try:
                result = SearchResult(
                    title=item.get("title", "No title"),
                    url=item.get("href") or item.get("link", ""),
                    snippet=item.get("body") or item.get("description", ""),
                    source=item.get("source"),
                )
                results.append(result)
            except Exception as e:
                logger.debug(f"Failed to parse search result: {e}")
                continue

        return SearchResponse(
            status="success",
            query_params=query_params,
            result_count=len(results),
            results=results,
        ).model_dump()

    except Exception as e:
        return SearchResponse(
            status="error",
            query_params=query_params,
            result_count=0,
            error_message=f"DuckDuckGo search failed: {type(e).__name__}: {str(e)}",
        ).model_dump()


# =============================================================================
# DDoS Secrets Search Implementation
# =============================================================================


async def search_ddos_secrets_db(
    query: str,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Search DDoS Secrets for leaked/hacked datasets.

    DDoS Secrets (Distributed Denial of Secrets) is a transparency collective
    that archives and publishes leaked data from government agencies, corporations,
    and other organizations. Use this to:
    - Find leaked documents and datasets
    - Research data breaches and hacks
    - Investigate government/corporate transparency issues
    - Access whistleblower materials

    WARNING: This tool uses web scraping as DDoS Secrets has no official API.
    Results may be incomplete or unavailable if the site structure changes.

    Args:
        query: Search keywords (e.g., "government surveillance", "police files")
        max_results: Maximum number of results to return (1-50). Default is 20.

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'
        - query_params: The parameters used for the query
        - result_count: Number of leaks found
        - results: List of leaked datasets with titles, URLs, and descriptions
        - error_message: Error details if the query failed

    Example:
        >>> result = await search_ddos_secrets_db(
        ...     query="police department",
        ...     max_results=10
        ... )
    """
    max_results = max(1, min(50, max_results))

    query_params = {
        "query": query,
        "max_results": max_results,
        "query_time": datetime.utcnow().isoformat(),
    }

    base_url = "https://ddosecrets.com"
    search_url = f"{base_url}/search"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Perform search request
            response = await client.get(
                search_url,
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; OSINTBot/1.0; +https://github.com/project-overwatch)"
                },
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "lxml")

            # Find search results
            # Note: This is based on current site structure and may break if they change it
            results: list[LeakResult] = []

            # Try to find article/leak elements
            # Common patterns: <article>, <div class="leak">, <div class="result">, etc.
            search_containers = (
                soup.find_all("article")
                or soup.find_all("div", class_=lambda x: x and "leak" in x.lower())
                or soup.find_all("div", class_=lambda x: x and "result" in x.lower())
            )

            if not search_containers:
                # Try alternative: look for links in the main content area
                main_content = soup.find("main") or soup.find("div", class_="content")
                if main_content:
                    # Find all links that look like dataset pages
                    links = main_content.find_all("a", href=True)
                    for link in links[:max_results]:
                        href = link.get("href", "")
                        if not href or href.startswith("#") or href.startswith("http"):
                            if href.startswith("http") and "ddosecrets.com" not in href:
                                continue

                        # Build full URL
                        full_url = href if href.startswith("http") else f"{base_url}{href}"

                        # Skip navigation/utility links
                        if any(
                            skip in href.lower()
                            for skip in [
                                "/about",
                                "/submit",
                                "/contact",
                                "/all_categories",
                                "/type/",
                                "/country/",
                                "/source/",
                            ]
                        ):
                            continue

                        result = LeakResult(
                            title=link.get_text(strip=True) or "Unknown",
                            url=full_url,
                            description=None,
                            category=None,
                            date=None,
                        )
                        results.append(result)

            else:
                # Parse structured search results
                for container in search_containers[:max_results]:
                    try:
                        # Try to find title
                        title_elem = (
                            container.find("h1")
                            or container.find("h2")
                            or container.find("h3")
                            or container.find("a")
                        )
                        title = (
                            title_elem.get_text(strip=True) if title_elem else "Unknown"
                        )

                        # Try to find URL
                        link_elem = container.find("a", href=True)
                        if not link_elem:
                            continue
                        href = link_elem.get("href", "")
                        full_url = href if href.startswith("http") else f"{base_url}{href}"

                        # Try to find description
                        desc_elem = container.find("p") or container.find(
                            "div", class_=lambda x: x and "desc" in x.lower()
                        )
                        description = desc_elem.get_text(strip=True) if desc_elem else None

                        # Try to find category/type
                        category_elem = container.find(
                            class_=lambda x: x and ("category" in x.lower() or "type" in x.lower())
                        )
                        category = category_elem.get_text(strip=True) if category_elem else None

                        # Try to find date
                        date_elem = container.find("time") or container.find(
                            class_=lambda x: x and "date" in x.lower()
                        )
                        date = date_elem.get_text(strip=True) if date_elem else None

                        result = LeakResult(
                            title=title,
                            url=full_url,
                            description=description,
                            category=category,
                            date=date,
                        )
                        results.append(result)

                    except Exception as e:
                        logger.debug(f"Failed to parse leak result: {e}")
                        continue

            if not results:
                return LeakSearchResponse(
                    status="success",
                    query_params=query_params,
                    result_count=0,
                    error_message="No results found. The search may be too specific or the site structure has changed.",
                ).model_dump()

            return LeakSearchResponse(
                status="success",
                query_params=query_params,
                result_count=len(results),
                results=results,
            ).model_dump()

    except httpx.TimeoutException:
        return LeakSearchResponse(
            status="error",
            query_params=query_params,
            result_count=0,
            error_message="Request to DDoS Secrets timed out. The site may be slow or unavailable.",
        ).model_dump()

    except httpx.HTTPStatusError as e:
        return LeakSearchResponse(
            status="error",
            query_params=query_params,
            result_count=0,
            error_message=f"DDoS Secrets returned HTTP {e.response.status_code}. The site may be down or blocking requests.",
        ).model_dump()

    except Exception as e:
        return LeakSearchResponse(
            status="error",
            query_params=query_params,
            result_count=0,
            error_message=f"Failed to search DDoS Secrets: {type(e).__name__}: {str(e)}. The site structure may have changed.",
        ).model_dump()
