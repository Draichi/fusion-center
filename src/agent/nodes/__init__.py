"""
Research Agent Nodes - Strategy Pattern Implementation.

Each node is a strategy for processing a specific phase of the research workflow.
All nodes inherit from BaseNode which provides common functionality via Template Method.
"""

from src.agent.nodes.base import BaseNode
from src.agent.nodes.planning import PlanningNode
from src.agent.nodes.decomposition import DecompositionNode
from src.agent.nodes.hypothesis import HypothesisGenerationNode, HypothesisUpdateNode
from src.agent.nodes.gathering import gather_intelligence
from src.agent.nodes.analysis import AnalysisNode
from src.agent.nodes.reflection import ReflectionNode
from src.agent.nodes.correlation import CorrelationNode
from src.agent.nodes.verification import VerificationNode
from src.agent.nodes.synthesis import SynthesisNode
from src.agent.nodes.router import route_next_step

__all__ = [
    "BaseNode",
    "PlanningNode",
    "DecompositionNode",
    "HypothesisGenerationNode",
    "HypothesisUpdateNode",
    "gather_intelligence",
    "AnalysisNode",
    "ReflectionNode",
    "CorrelationNode",
    "VerificationNode",
    "SynthesisNode",
    "route_next_step",
]

