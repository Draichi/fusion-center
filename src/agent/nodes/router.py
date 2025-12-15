"""
Router function for determining the next node in the graph.
"""

from typing import Literal

from src.agent.state import AgentState, ResearchPhase
from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


def route_next_step(state: AgentState) -> Literal[
    "decompose", "hypothesize", "gather", "analyze", "reflect", "correlate", "verify", "synthesize", "end"
]:
    """
    Determine the next node based on current state.
    
    This is used as a conditional edge in the graph.
    Supports both traditional and multi-step reasoning phases.
    """
    phase = state.get("current_phase", ResearchPhase.PLANNING.value)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", settings.agent_max_iterations)
    
    # Safety check for max iterations
    if iteration >= max_iter:
        logger.warning(f"Max iterations ({max_iter}) reached, forcing synthesis")
        return "synthesize"
    
    # Check for errors
    if state.get("error"):
        return "end"
    
    # Route based on phase (including new multi-step reasoning phases)
    if phase == ResearchPhase.DECOMPOSING.value:
        return "decompose"
    elif phase == ResearchPhase.HYPOTHESIZING.value:
        return "hypothesize"
    elif phase == ResearchPhase.GATHERING.value:
        return "gather"
    elif phase == ResearchPhase.ANALYZING.value:
        return "analyze"
    elif phase == ResearchPhase.REFLECTING.value:
        return "reflect"
    elif phase == ResearchPhase.CORRELATING.value:
        return "correlate"
    elif phase == ResearchPhase.VERIFYING.value:
        return "verify"
    elif phase == ResearchPhase.SYNTHESIZING.value:
        return "synthesize"
    elif phase == ResearchPhase.COMPLETE.value:
        return "end"
    else:
        return "end"

