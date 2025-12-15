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
    # LLM Provider API Keys
    # ==========================================================================
    
    # Google Gemini
    google_api_key: str | None = None
    
    # xAI Grok
    xai_api_key: str | None = None
    
    # Ollama (local - no key needed, just URL)
    ollama_base_url: str = "http://localhost:11434"
    
    # Docker Model Runner (local - no key needed, OpenAI-compatible API)
    docker_model_base_url: str = "http://localhost:12434/engines/llama.cpp/v1"

    # ==========================================================================
    # Agent Configuration
    # ==========================================================================
    
    # Available providers: gemini, grok, ollama, docker
    agent_provider: Literal["gemini", "grok", "ollama", "docker"] = "gemini"
    
    # Model names per provider
    agent_model_gemini: str = "gemini-2.0-flash-exp"
    agent_model_grok: str = "grok-beta"
    agent_model_ollama: str = "llama3.2"
    agent_model_docker: str = "ai/qwen3:4B-UD-Q4_K_XL"
    
    agent_temperature: float = 0.7
    agent_max_iterations: int = 10

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

    @property
    def has_google_key(self) -> bool:
        """Check if Google API key is configured."""
        return bool(self.google_api_key)

    @property
    def has_xai_key(self) -> bool:
        """Check if xAI (Grok) API key is configured."""
        return bool(self.xai_api_key)

    @property
    def agent_model(self) -> str:
        """Get the model name for the configured provider."""
        models = {
            "gemini": self.agent_model_gemini,
            "grok": self.agent_model_grok,
            "ollama": self.agent_model_ollama,
            "docker": self.agent_model_docker,
        }
        return models.get(self.agent_provider, self.agent_model_gemini)

    def get_available_providers(self) -> list[str]:
        """Get list of providers with configured API keys."""
        available = []
        if self.has_google_key:
            available.append("gemini")
        if self.has_xai_key:
            available.append("grok")
        # Ollama is always available (local)
        available.append("ollama")
        # Docker Model Runner is always available (local)
        available.append("docker")
        return available


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
