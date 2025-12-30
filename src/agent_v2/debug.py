"""
Prompt Debug Utility for Agent v2.

When DEBUG_PROMPTS=1 is set, logs all prompts to files in logs/ directory.
"""

import os
from datetime import datetime
from pathlib import Path


DEBUG_PROMPTS = os.getenv("DEBUG_PROMPTS", "").lower() in ("1", "true", "yes")

_log_dir: Path | None = None
_run_id: str | None = None


def _ensure_log_dir() -> Path:
    """Create and return the log directory for this run."""
    global _log_dir, _run_id
    
    if _log_dir is None:
        _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_dir = Path("logs") / f"prompts_{_run_id}"
        _log_dir.mkdir(parents=True, exist_ok=True)
    
    return _log_dir


def log_prompt(
    phase: str,
    prompt: str,
    iteration: int = 0,
    extra_context: str = "",
) -> None:
    """
    Log a prompt to a file if DEBUG_PROMPTS is enabled.
    
    Args:
        phase: The research phase (e.g., "planning", "analysis")
        prompt: The actual prompt text
        iteration: The iteration number (for gather/analyze/reflect loop)
        extra_context: Any additional context to log
    """
    if not DEBUG_PROMPTS:
        return
    
    log_dir = _ensure_log_dir()
    
    # Create filename
    if iteration > 0:
        filename = f"{phase}_iter{iteration}.txt"
    else:
        filename = f"{phase}.txt"
    
    filepath = log_dir / filename
    
    # Write prompt to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Phase: {phase}\n")
        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
        if iteration > 0:
            f.write(f"# Iteration: {iteration}\n")
        if extra_context:
            f.write(f"# Context: {extra_context}\n")
        f.write(f"# {'=' * 60}\n\n")
        f.write(prompt)
    
    from src.shared.logger import get_logger
    logger = get_logger()
    logger.debug(f"📝 Prompt logged to: {filepath}")


def get_log_dir() -> Path | None:
    """Get the current log directory, if any."""
    return _log_dir


def get_run_id() -> str | None:
    """Get the current run ID, if any."""
    return _run_id
