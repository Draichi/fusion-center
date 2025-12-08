"""
Project Overwatch - Deep Research Agent Module.

This module provides an autonomous AI agent for OSINT and geopolitical intelligence
analysis using LangGraph for orchestration.

Example:
    ```python
    from src.agent import DeepResearchAgent
    
    agent = DeepResearchAgent()
    result = await agent.research("Analyze military activity in Ukraine")
    print(result["executive_summary"])
    ```
"""

__version__ = "0.1.0"

from src.agent.core import (
    DeepResearchAgent,
    run_deep_research,
    AgentState,
    ResearchPhase,
    IntelligenceType,
)

__all__ = [
    "DeepResearchAgent",
    "run_deep_research",
    "AgentState",
    "ResearchPhase",
    "IntelligenceType",
]
