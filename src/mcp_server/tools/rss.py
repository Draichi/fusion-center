"""
RSS Feed Integration Module.

This module provides tools to fetch and parse RSS feeds from independent news sources
including Meduza, The Insider, and The Cradle.
"""

import re
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.shared.logger import get_logger

logger = get_logger()


# RSS Feed URLs
RSS_FEEDS = {
    "meduza": {
        "name": "Meduza",
        "url": "https://meduza.io/rss/all",
        "description": "Independent Russian news source",
    },
    "theinsider": {
        "name": "The Insider",
        "url": "https://theinsider.me/feed/",
        "description": "Independent Russian investigative journalism",
    },
    "thecradle": {
        "name": "The Cradle",
        "url": "https://thecradle.co/feed",
        "description": "Geopolitical news covering West Asia",
    },
}


class RSSArticle(BaseModel):
    """Represents an article from an RSS feed."""

    title: str = Field(description="Article title")
    link: str = Field(description="Article URL")
    published: str = Field(description="Publication date")
    description: str | None = Field(default=None, description="Article description/summary")
    source: str = Field(description="Source name (e.g., Meduza, The Insider)")
    feed_url: str = Field(description="RSS feed URL")


class RSSResponse(BaseModel):
    """Response model for RSS feed queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    source: str = Field(description="Source name")
    feed_url: str = Field(description="RSS feed URL")
    article_count: int = Field(description="Number of articles found")
    articles: list[RSSArticle] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="RSS Feed", description="Data source identifier")


def _parse_rss_xml(xml_content: str, source_name: str, feed_url: str) -> RSSResponse:
    """
    Parse RSS XML content and extract articles.
    
    Handles both RSS 2.0 and Atom formats, with namespace support.
    
    Args:
        xml_content: Raw XML content from RSS feed
        source_name: Name of the source
        feed_url: URL of the RSS feed
        
    Returns:
        RSSResponse with parsed articles
    """
    import xml.etree.ElementTree as ET
    
    articles: list[RSSArticle] = []
    
    try:
        # Parse XML - handle potential encoding issues
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            # Try with explicit UTF-8 encoding declaration
            if not xml_content.strip().startswith("<?xml"):
                xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_content
            root = ET.fromstring(xml_content)
        
        # Define namespaces
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "content": "http://purl.org/rss/1.0/modules/content/",
        }
        
        # Try RSS 2.0 format first
        channel = root.find("channel")
        if channel is not None:
            items = channel.findall("item")
            for item in items:
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_date_elem = item.find("pubDate")
                description_elem = item.find("description")
                
                # Try content:encoded for full description
                if description_elem is None:
                    description_elem = item.find("content:encoded", namespaces)
                
                if title_elem is not None and link_elem is not None:
                    title = (title_elem.text or "").strip()
                    link = (link_elem.text or "").strip()
                    published = (pub_date_elem.text or "").strip() if pub_date_elem is not None else ""
                    
                    # Clean description - remove HTML tags if present
                    description = None
                    if description_elem is not None and description_elem.text:
                        desc_text = description_elem.text.strip()
                        # Simple HTML tag removal
                        desc_text = re.sub(r"<[^>]+>", "", desc_text)
                        description = desc_text if desc_text else None
                    
                    if title and link:
                        articles.append(
                            RSSArticle(
                                title=title,
                                link=link,
                                published=published,
                                description=description,
                                source=source_name,
                                feed_url=feed_url,
                            )
                        )
        else:
            # Try Atom format
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                # Try without namespace
                entries = root.findall(".//entry")
            
            for entry in entries:
                # Try with namespace first
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_elem is None:
                    title_elem = entry.find("title")
                
                link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                if link_elem is None:
                    link_elem = entry.find("link")
                
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                if published_elem is None:
                    published_elem = entry.find("published")
                
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                if summary_elem is None:
                    summary_elem = entry.find("summary")
                
                if title_elem is not None and link_elem is not None:
                    title = (title_elem.text or "").strip()
                    # Atom links can be in href attribute or text
                    link = link_elem.get("href", "") if link_elem.get("href") else (link_elem.text or "").strip()
                    published = (published_elem.text or "").strip() if published_elem is not None else ""
                    
                    description = None
                    if summary_elem is not None and summary_elem.text:
                        desc_text = summary_elem.text.strip()
                        desc_text = re.sub(r"<[^>]+>", "", desc_text)
                        description = desc_text if desc_text else None
                    
                    if title and link:
                        articles.append(
                            RSSArticle(
                                title=title,
                                link=link,
                                published=published,
                                description=description,
                                source=source_name,
                                feed_url=feed_url,
                            )
                        )
        
        return RSSResponse(
            status="success",
            source=source_name,
            feed_url=feed_url,
            article_count=len(articles),
            articles=articles,
        )
        
    except ET.ParseError as e:
        logger.error(f"Failed to parse RSS XML: {e}")
        return RSSResponse(
            status="error",
            source=source_name,
            feed_url=feed_url,
            article_count=0,
            error_message=f"Failed to parse RSS XML: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error parsing RSS: {e}")
        return RSSResponse(
            status="error",
            source=source_name,
            feed_url=feed_url,
            article_count=0,
            error_message=f"Unexpected error parsing RSS: {type(e).__name__}: {str(e)}",
        )


async def fetch_rss_feed(
    source: str,
    max_articles: int = 20,
) -> dict[str, Any]:
    """
    Fetch and parse articles from an RSS feed.
    
    Supported sources:
    - meduza: Meduza (independent Russian news)
    - theinsider: The Insider (Russian investigative journalism)
    - thecradle: The Cradle (geopolitical news covering West Asia)
    
    Args:
        source: Source identifier (meduza, theinsider, or thecradle)
        max_articles: Maximum number of articles to return (1-50). Default: 20.
        
    Returns:
        A dictionary containing:
        - status: 'success' or 'error'
        - source: Source name
        - feed_url: RSS feed URL
        - article_count: Number of articles found
        - articles: List of articles with titles, URLs, dates, and descriptions
        - error_message: Error details if the query failed
        
    Example:
        >>> result = await fetch_rss_feed(
        ...     source="meduza",
        ...     max_articles=10
        ... )
    """
    source_lower = source.lower().strip()
    
    if source_lower not in RSS_FEEDS:
        available = ", ".join(RSS_FEEDS.keys())
        return RSSResponse(
            status="error",
            source=source,
            feed_url="",
            article_count=0,
            error_message=f"Unknown source '{source}'. Available sources: {available}",
        ).model_dump()
    
    feed_info = RSS_FEEDS[source_lower]
    feed_url = feed_info["url"]
    source_name = feed_info["name"]
    
    max_articles = max(1, min(50, max_articles))
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            xml_content = response.text
            
            if not xml_content or not xml_content.strip():
                return RSSResponse(
                    status="error",
                    source=source_name,
                    feed_url=feed_url,
                    article_count=0,
                    error_message="RSS feed returned empty content",
                ).model_dump()
            
            # Parse RSS XML
            result = _parse_rss_xml(xml_content, source_name, feed_url)
            
            # Limit number of articles
            if len(result.articles) > max_articles:
                result.articles = result.articles[:max_articles]
                result.article_count = len(result.articles)
            
            return result.model_dump()
            
    except httpx.TimeoutException:
        return RSSResponse(
            status="error",
            source=source_name,
            feed_url=feed_url,
            article_count=0,
            error_message=f"Request to RSS feed timed out. Try again later.",
        ).model_dump()
    
    except httpx.HTTPStatusError as e:
        return RSSResponse(
            status="error",
            source=source_name,
            feed_url=feed_url,
            article_count=0,
            error_message=f"RSS feed returned HTTP {e.response.status_code}: {e.response.text[:200]}",
        ).model_dump()
    
    except httpx.ConnectError as e:
        return RSSResponse(
            status="error",
            source=source_name,
            feed_url=feed_url,
            article_count=0,
            error_message=f"Failed to connect to RSS feed: {str(e)}. Check network connectivity.",
        ).model_dump()
    
    except Exception as e:
        return RSSResponse(
            status="error",
            source=source_name,
            feed_url=feed_url,
            article_count=0,
            error_message=f"Unexpected error fetching RSS feed: {type(e).__name__}: {str(e)}",
        ).model_dump()
