"""
Centralized configuration for Project Overwatch.

Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Configuration
    mcp_server_name: str = "project-overwatch"
    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8080
    
    # Dashboard Configuration
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    
    # Event Correlation Engine Configuration
    ece_db_path: str = "data/ece.db"

    # ==========================================================================
    # Data Source API Keys
    # ==========================================================================
    nasa_firms_api_key: str | None = None
    cloudflare_api_token: str | None = None
    
    # Telegram API credentials (obtain from https://my.telegram.org)
    telegram_api_id: int | None = None
    telegram_api_hash: str | None = None
    
    # AlienVault OTX API key (obtain from https://otx.alienvault.com)
    otx_api_key: str | None = None

    # External API URLs
    gdelt_api_base_url: str = "https://api.gdeltproject.org/api/v2"

    # ==========================================================================
    # LLM Provider Configuration (Ollama only)
    # ==========================================================================
    
    # Ollama (local - no key needed)
    # Note: /v1 suffix is required for OpenAI-compatible API
    ollama_base_url: str = "http://localhost:11434/v1"

    # ==========================================================================
    # Agent Configuration
    # ==========================================================================
    
    agent_model: str = "llama3.2"
    agent_temperature: float = 0.7
    agent_max_iterations: int = 10

    # ==========================================================================
    # Dual LLM Configuration
    # ==========================================================================
    
    # Structured Output LLM (JSON mode, for planning/analysis/correlation nodes)
    structured_llm_model: str = "qwen2.5:7b"
    structured_llm_temperature: float = 0.3
    
    # Thinking/Reasoning LLM (for reflection/verification/synthesis nodes)
    thinking_llm_model: str = "llama3.2"
    thinking_llm_temperature: float = 0.7

    # Logging
    log_level: str = "INFO"

    # Output Configuration
    output_dir: str = "output"
    save_report: bool = True
    save_reasoning_log: bool = True

    # ==========================================================================
    # Computed Properties
    # ==========================================================================

    @property
    def has_nasa_key(self) -> bool:
        """Check if NASA FIRMS API key is configured."""
        return bool(self.nasa_firms_api_key)

    @property
    def has_telegram_credentials(self) -> bool:
        """Check if Telegram API credentials are configured."""
        return bool(self.telegram_api_id and self.telegram_api_hash)

    @property
    def has_otx_key(self) -> bool:
        """Check if AlienVault OTX API key is configured."""
        return bool(self.otx_api_key)



@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
