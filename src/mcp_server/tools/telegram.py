"""
Telegram Channel Monitoring Module.

This module provides tools to monitor public Telegram channels for OSINT purposes,
particularly useful for tracking real-time information from conflict zones and
news sources that primarily operate on Telegram.

Requires Telegram API credentials:
- TELEGRAM_API_ID: Obtain from https://my.telegram.org
- TELEGRAM_API_HASH: Obtain from https://my.telegram.org
"""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()

# Telethon imports - handled gracefully if not available
try:
    from telethon import TelegramClient
    from telethon.tl.functions.messages import GetHistoryRequest
    from telethon.tl.types import Channel, Message
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False
    TelegramClient = None
    GetHistoryRequest = None
    Channel = None
    Message = None


# =============================================================================
# Data Models
# =============================================================================


class TelegramMessage(BaseModel):
    """Represents a message from a Telegram channel."""

    message_id: int = Field(description="Unique message ID")
    channel_name: str = Field(description="Name of the source channel")
    channel_username: str = Field(description="Username of the channel (without @)")
    text: str = Field(description="Message text content")
    date: str = Field(description="Message date in ISO format")
    views: int | None = Field(default=None, description="Number of views")
    forwards: int | None = Field(default=None, description="Number of forwards")
    has_media: bool = Field(default=False, description="Whether message contains media")
    media_type: str | None = Field(default=None, description="Type of media if present")
    url: str = Field(description="Direct link to the message")


class TelegramResponse(BaseModel):
    """Response model for Telegram channel queries."""

    status: str = Field(description="Query status: 'success' or 'error'")
    query_params: dict[str, Any] = Field(description="Parameters used in the query")
    message_count: int = Field(description="Number of messages found")
    messages: list[TelegramMessage] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Error message if any")
    data_source: str = Field(default="Telegram", description="Data source identifier")


# =============================================================================
# Curated OSINT Channels
# =============================================================================

# =============================================================================
# Curated OSINT Channels
# =============================================================================
# PUBLIC channels for conflict monitoring.
# 
# ⚠️  IMPORTANT: Channel names change frequently!
# Verify channels exist at: https://t.me/{channel_name}
# Update this list with your own trusted channels.
#
# TIP: Use the 'channels' parameter to search specific channels directly
# without modifying this list.
# =============================================================================

OSINT_CHANNELS = {
    # News channels (independent journalism)
    "news": [
        "meduzalive",
        "theinsider",
        "thebell_io",
        "KyivIndependent_official",
    ],
    # OSINT aggregators and analysis
    "osint_general": [
        "rybar",
        "bellingcat",
    ],
}

# Flatten all channels for easy lookup
ALL_OSINT_CHANNELS = []
for category_channels in OSINT_CHANNELS.values():
    ALL_OSINT_CHANNELS.extend(category_channels)


# =============================================================================
# Telegram Client Management
# =============================================================================


def _get_session_path() -> Path:
    """Get the path for storing Telegram session files."""
    # Store session in project directory for portability
    # Find project root by looking for pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            session_dir = parent / ".telegram_session"
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir / "session"
    
    # Fallback to home directory if project root not found
    session_dir = Path.home() / ".overwatch" / "telegram"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir / "overwatch_session"


async def _get_client() -> "TelegramClient | None":
    """
    Get an authenticated Telegram client.
    
    Returns None if credentials are not configured or Telethon is not available.
    """
    if not TELETHON_AVAILABLE:
        return None
    
    api_id = settings.telegram_api_id
    api_hash = settings.telegram_api_hash
    
    if not api_id or not api_hash:
        return None
    
    session_path = str(_get_session_path())
    client = TelegramClient(session_path, api_id, api_hash)
    
    await client.start()
    return client


# =============================================================================
# Main Tool Functions
# =============================================================================


async def search_telegram_channels(
    keywords: str | None = None,
    channels: list[str] | None = None,
    category: str | None = None,
    hours_back: int = 24,
    max_messages: int = 50,
) -> dict[str, Any]:
    """
    Search public Telegram channels for messages matching criteria.

    This tool monitors public Telegram channels for OSINT purposes, particularly
    useful for tracking real-time information from:
    - Conflict zones (Ukraine/Russia, Middle East, etc.)
    - Independent news sources operating on Telegram
    - Military and defense analysis channels
    - Breaking news before it reaches mainstream media

    Args:
        keywords: Optional search terms to filter messages. Supports simple
                  text matching (case-insensitive). Examples: "missile", 
                  "Kharkiv", "drone strike". Leave empty to get all recent messages.
        channels: Optional list of specific channel usernames to search
                  (without @ symbol). Examples: ["ukrainenowenglish", "ryaborov"].
                  If not provided, searches curated OSINT channels.
        category: Optional category of curated channels to search:
                  - "news": Independent news channels (Meduza, The Insider, etc.)
                  - "osint_general": OSINT aggregators and analysis (Rybar, Bellingcat)
                  Leave empty to search all categories.
        hours_back: Number of hours to look back (1-168). Default: 24 hours.
        max_messages: Maximum messages to return per channel (1-100). Default: 50.

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - query_params: Parameters used for the query
        - message_count: Number of messages found
        - messages: List of messages with text, timestamps, and metadata
        - error_message: Error details if the query failed

    Example:
        >>> result = await search_telegram_channels(
        ...     keywords="missile strike",
        ...     category="ukraine_conflict",
        ...     hours_back=12
        ... )
    """
    # Validate parameters
    hours_back = max(1, min(168, hours_back))
    max_messages = max(1, min(100, max_messages))
    
    query_params = {
        "keywords": keywords,
        "channels": channels,
        "category": category,
        "hours_back": hours_back,
        "max_messages": max_messages,
        "query_time": datetime.now(timezone.utc).isoformat(),
    }

    # Check if Telethon is available
    if not TELETHON_AVAILABLE:
        return TelegramResponse(
            status="error",
            query_params=query_params,
            message_count=0,
            error_message="TOOL DISABLED: Telethon library is not installed. "
            "Install it with: pip install telethon",
        ).model_dump()

    # Check credentials
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        return TelegramResponse(
            status="error",
            query_params=query_params,
            message_count=0,
            error_message="TOOL DISABLED: Telegram API credentials not configured. "
            "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in your .env file. "
            "Get credentials at: https://my.telegram.org",
        ).model_dump()

    # Determine which channels to search
    target_channels: list[str] = []
    
    if channels:
        # Use explicitly provided channels
        target_channels = [c.lstrip("@") for c in channels]
    elif category:
        # Use channels from specified category
        if category in OSINT_CHANNELS:
            target_channels = OSINT_CHANNELS[category]
        else:
            return TelegramResponse(
                status="error",
                query_params=query_params,
                message_count=0,
                error_message=f"Unknown category: '{category}'. "
                f"Available: {list(OSINT_CHANNELS.keys())}",
            ).model_dump()
    else:
        # Search all curated channels
        target_channels = ALL_OSINT_CHANNELS

    # Calculate time threshold
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    
    all_messages: list[TelegramMessage] = []
    errors: list[str] = []
    
    try:
        client = await _get_client()
        if not client:
            return TelegramResponse(
                status="error",
                query_params=query_params,
                message_count=0,
                error_message="Failed to initialize Telegram client. Check credentials.",
            ).model_dump()
        
        async with client:
            for channel_username in target_channels:
                try:
                    # Get channel entity
                    try:
                        entity = await client.get_entity(channel_username)
                    except Exception as e:
                        logger.warning(f"Could not access channel @{channel_username}: {e}")
                        errors.append(f"@{channel_username}: access denied or not found")
                        continue
                    
                    # Get channel name
                    channel_name = getattr(entity, 'title', channel_username)
                    
                    # Fetch recent messages
                    messages = await client.get_messages(
                        entity,
                        limit=max_messages,
                        offset_date=datetime.now(timezone.utc),
                    )
                    
                    for msg in messages:
                        if not msg or not msg.date:
                            continue
                        
                        # Check if message is within time range
                        msg_date = msg.date.replace(tzinfo=timezone.utc) if msg.date.tzinfo is None else msg.date
                        if msg_date < time_threshold:
                            continue
                        
                        # Get message text
                        text = msg.text or msg.message or ""
                        if not text:
                            continue
                        
                        # Filter by keywords if provided
                        if keywords:
                            keyword_pattern = re.compile(
                                re.escape(keywords), 
                                re.IGNORECASE
                            )
                            if not keyword_pattern.search(text):
                                continue
                        
                        # Determine media type
                        has_media = msg.media is not None
                        media_type = None
                        if has_media:
                            media_class = type(msg.media).__name__
                            media_type = media_class.replace("MessageMedia", "").lower()
                        
                        # Create message object
                        telegram_msg = TelegramMessage(
                            message_id=msg.id,
                            channel_name=channel_name,
                            channel_username=channel_username,
                            text=text[:2000],  # Truncate very long messages
                            date=msg_date.isoformat(),
                            views=getattr(msg, 'views', None),
                            forwards=getattr(msg, 'forwards', None),
                            has_media=has_media,
                            media_type=media_type,
                            url=f"https://t.me/{channel_username}/{msg.id}",
                        )
                        all_messages.append(telegram_msg)
                    
                    logger.debug(f"Fetched {len(messages)} messages from @{channel_username}")
                    
                except Exception as e:
                    logger.warning(f"Error fetching from @{channel_username}: {e}")
                    errors.append(f"@{channel_username}: {str(e)}")
                    continue
        
        # Sort messages by date (newest first)
        all_messages.sort(key=lambda m: m.date, reverse=True)
        
        # Build response
        error_msg = None
        if errors and not all_messages:
            error_msg = f"Could not fetch from any channels. Errors: {'; '.join(errors)}"
        elif errors:
            error_msg = f"Partial results. Some channels failed: {'; '.join(errors[:3])}"
        
        return TelegramResponse(
            status="success" if all_messages else "error",
            query_params=query_params,
            message_count=len(all_messages),
            messages=all_messages,
            error_message=error_msg,
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Telegram search failed: {e}")
        return TelegramResponse(
            status="error",
            query_params=query_params,
            message_count=0,
            error_message=f"Unexpected error: {type(e).__name__}: {str(e)}",
        ).model_dump()


async def get_telegram_channel_info(
    channel_username: str,
) -> dict[str, Any]:
    """
    Get information about a specific Telegram channel.

    Retrieves metadata about a public Telegram channel including:
    - Channel name and description
    - Subscriber count
    - Whether it's verified
    - Creation date (approximate)

    Args:
        channel_username: The channel username (with or without @).
                          Example: "ukrainenowenglish" or "@ukrainenowenglish"

    Returns:
        Dictionary containing channel information or error details.

    Example:
        >>> result = await get_telegram_channel_info("ryaborov")
    """
    channel_username = channel_username.lstrip("@")
    
    query_params = {
        "channel_username": channel_username,
        "query_time": datetime.now(timezone.utc).isoformat(),
    }

    if not TELETHON_AVAILABLE:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": "Telethon library not installed.",
        }

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": "Telegram API credentials not configured.",
        }

    try:
        client = await _get_client()
        if not client:
            return {
                "status": "error",
                "query_params": query_params,
                "error_message": "Failed to initialize Telegram client.",
            }
        
        async with client:
            try:
                entity = await client.get_entity(channel_username)
            except Exception as e:
                return {
                    "status": "error",
                    "query_params": query_params,
                    "error_message": f"Could not find channel @{channel_username}: {str(e)}",
                }
            
            # Extract channel info
            full_info = await client.get_entity(entity)
            
            # Get participant count if available
            participants_count = None
            try:
                if hasattr(full_info, 'participants_count'):
                    participants_count = full_info.participants_count
            except Exception:
                pass
            
            return {
                "status": "success",
                "query_params": query_params,
                "channel_info": {
                    "username": channel_username,
                    "title": getattr(full_info, 'title', channel_username),
                    "description": getattr(full_info, 'about', None),
                    "participants_count": participants_count,
                    "verified": getattr(full_info, 'verified', False),
                    "scam": getattr(full_info, 'scam', False),
                    "fake": getattr(full_info, 'fake', False),
                    "url": f"https://t.me/{channel_username}",
                },
                "data_source": "Telegram",
            }
            
    except Exception as e:
        return {
            "status": "error",
            "query_params": query_params,
            "error_message": f"Error fetching channel info: {str(e)}",
        }


def list_curated_channels() -> dict[str, Any]:
    """
    List all curated OSINT channels organized by category.

    Returns a categorized list of public Telegram channels that are
    monitored for OSINT purposes. These channels cover various perspectives
    and topics related to geopolitical events.

    Returns:
        Dictionary with channel categories and their descriptions.
    """
    return {
        "status": "success",
        "categories": {
            "ukraine_conflict": {
                "description": "Pro-Ukraine channels covering the conflict",
                "channels": OSINT_CHANNELS["ukraine_conflict"],
            },
            "russia_perspective": {
                "description": "Russian perspective and military analysis",
                "channels": OSINT_CHANNELS["russia_perspective"],
            },
            "osint_general": {
                "description": "General OSINT aggregators and analysis",
                "channels": OSINT_CHANNELS["osint_general"],
            },
            "news": {
                "description": "News outlets with Telegram presence",
                "channels": OSINT_CHANNELS["news"],
            },
        },
        "total_channels": len(ALL_OSINT_CHANNELS),
        "note": "All channels are public and can be accessed without special permissions.",
    }

