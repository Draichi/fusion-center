"""
Project Overwatch - Deep Research Agent Core.

This module re-exports the main agent components for easy importing.
"""

from src.agent.graph import DeepResearchAgent, run_deep_research
from src.agent.state import (
    AgentState,
    Finding,
    Correlation,
    ResearchPlan,
    ResearchPhase,
    IntelligenceType,
    # Multi-step Reasoning types
    HypothesisStatus,
    Hypothesis,
    ReasoningStep,
    SubTask,
    ReflectionNote,
    VerificationResult,
    create_initial_state,
)
from src.agent.tools import MCPToolExecutor, get_tool_definitions

__all__ = [
    # Main Agent
    "DeepResearchAgent",
    "run_deep_research",
    # State
    "AgentState",
    "Finding",
    "Correlation",
    "ResearchPlan",
    "ResearchPhase",
    "IntelligenceType",
    "create_initial_state",
    # Multi-step Reasoning types
    "HypothesisStatus",
    "Hypothesis",
    "ReasoningStep",
    "SubTask",
    "ReflectionNote",
    "VerificationResult",
    # Tools
    "MCPToolExecutor",
    "get_tool_definitions",
]


# Backwards compatibility alias
OverwatchAgent = DeepResearchAgent
