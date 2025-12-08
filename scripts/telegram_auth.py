#!/usr/bin/env python3
"""
Telegram Authentication Script.

Run this script ONCE to authenticate your Telegram account.
After authentication, a session file is saved and the MCP tools
will work without requiring interactive login.

Usage:
    python scripts/telegram_auth.py

Requirements:
    - TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env
    - You'll need access to your phone to receive the auth code
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.shared.config import settings


async def main():
    """Authenticate with Telegram and create session file."""
    
    print("\n" + "="*60)
    print("🔐 Telegram Authentication Setup")
    print("="*60 + "\n")
    
    # Check credentials
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("❌ Error: Telegram credentials not configured!\n")
        print("Please set in your .env file:")
        print("  TELEGRAM_API_ID=your_api_id")
        print("  TELEGRAM_API_HASH=your_api_hash")
        print("\nGet credentials at: https://my.telegram.org")
        print("  1. Log in with your phone number")
        print("  2. Go to 'API development tools'")
        print("  3. Create a new application")
        return 1
    
    print(f"✓ API ID configured: {settings.telegram_api_id}")
    print(f"✓ API Hash configured: {settings.telegram_api_hash[:8]}...")
    
    # Import Telethon
    try:
        from telethon import TelegramClient
    except ImportError:
        print("\n❌ Error: Telethon not installed!")
        print("Run: pip install telethon")
        return 1
    
    # Session path - store in project directory
    project_root = Path(__file__).parent.parent
    session_dir = project_root / ".telegram_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "session"
    
    print(f"\n📁 Session will be saved to: {session_path}.session")
    
    # Check if session already exists
    if session_path.with_suffix(".session").exists():
        print("\n⚠️  Session file already exists!")
        response = input("Do you want to re-authenticate? [y/N]: ").strip().lower()
        if response != 'y':
            print("Keeping existing session. Exiting.")
            return 0
    
    print("\n" + "-"*60)
    print("Starting authentication...")
    print("You will need to enter your phone number and the code")
    print("sent to your Telegram app.")
    print("-"*60 + "\n")
    
    # Create client and authenticate
    client = TelegramClient(
        str(session_path),
        settings.telegram_api_id,
        settings.telegram_api_hash
    )
    
    await client.start()
    
    # Verify authentication
    me = await client.get_me()
    
    print("\n" + "="*60)
    print("✅ Authentication successful!")
    print("="*60)
    print(f"\n👤 Logged in as: {me.first_name} {me.last_name or ''}")
    print(f"📱 Phone: {me.phone}")
    print(f"🆔 User ID: {me.id}")
    print(f"\n📁 Session saved to: {session_path}.session")
    print("\n✨ You can now use the Telegram tools in Project Overwatch!")
    print("   The session will be reused automatically.\n")
    
    await client.disconnect()
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

