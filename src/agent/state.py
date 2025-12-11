"""
Agent State Schema for LangGraph.

Defines the state that flows through the research agent graph.
Includes Multi-step Reasoning support for deeper analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from src.shared.config import settings


class ResearchPhase(str, Enum):
    """Current phase of the research process."""
    
    PLANNING = "planning"
    DECOMPOSING = "decomposing"      # Break complex tasks into sub-tasks
    HYPOTHESIZING = "hypothesizing"  # Form hypotheses before gathering
    GATHERING = "gathering"
    ANALYZING = "analyzing"
    REFLECTING = "reflecting"        # Self-critique and review
    CORRELATING = "correlating"
    VERIFYING = "verifying"          # Verify consistency before synthesis
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"


class HypothesisStatus(str, Enum):
    """Status of a hypothesis in the reasoning process."""
    
    PROPOSED = "proposed"       # Initial hypothesis
    INVESTIGATING = "investigating"  # Currently gathering evidence
    SUPPORTED = "supported"     # Evidence supports it
    REFUTED = "refuted"        # Evidence contradicts it
    INCONCLUSIVE = "inconclusive"  # Not enough evidence


class IntelligenceType(str, Enum):
    """Types of intelligence being gathered."""
    
    NEWS = "news"
    SATELLITE = "satellite"
    CYBER = "cyber"
    SANCTIONS = "sanctions"
    COMBINED = "combined"


@dataclass
class Finding:
    """A single piece of intelligence finding."""
    
    source: str
    source_type: IntelligenceType
    timestamp: str
    content: dict[str, Any]
    relevance_score: float = 0.0
    confidence: str = "medium"
    location: dict[str, float] | None = None  # lat/lon if applicable
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "timestamp": self.timestamp,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "confidence": self.confidence,
            "location": self.location,
        }


@dataclass
class Correlation:
    """A correlation between multiple findings."""
    
    finding_ids: list[int]
    correlation_type: str
    description: str
    confidence: str
    implications: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_ids": self.finding_ids,
            "correlation_type": self.correlation_type,
            "description": self.description,
            "confidence": self.confidence,
            "implications": self.implications,
        }


@dataclass
class ResearchPlan:
    """A plan for conducting research."""
    
    objectives: list[str]
    regions_of_interest: list[str]
    keywords: list[str]
    coordinates: list[dict[str, float]]  # List of {lat, lon, radius_km}
    time_range: str
    priority_sources: list[IntelligenceType]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": self.objectives,
            "regions_of_interest": self.regions_of_interest,
            "keywords": self.keywords,
            "coordinates": self.coordinates,
            "time_range": self.time_range,
            "priority_sources": [s.value for s in self.priority_sources],
        }


@dataclass
class Hypothesis:
    """A hypothesis to be tested during research."""
    
    id: str
    statement: str
    status: HypothesisStatus
    confidence: float  # 0.0 to 1.0
    supporting_evidence: list[int] = field(default_factory=list)  # Finding indices
    contradicting_evidence: list[int] = field(default_factory=list)
    reasoning: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "reasoning": self.reasoning,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ReasoningStep:
    """A single step in the multi-step reasoning process."""
    
    step_number: int
    phase: str
    thought: str  # What the agent is thinking
    action: str   # What action it decided to take
    observation: str  # What it observed/learned
    conclusion: str  # What it concluded
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "phase": self.phase,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class SubTask:
    """A sub-task decomposed from the main research task."""
    
    id: str
    description: str
    parent_task: str | None  # ID of parent sub-task, None if top-level
    status: str  # pending, in_progress, completed
    findings_ids: list[int] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # IDs of sub-tasks this depends on
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "parent_task": self.parent_task,
            "status": self.status,
            "findings_ids": self.findings_ids,
            "dependencies": self.dependencies,
        }


@dataclass
class ReflectionNote:
    """A note from the agent's self-reflection process."""
    
    category: str  # bias_check, gap_analysis, alternative_explanation, confidence_calibration
    content: str
    severity: str  # info, warning, critical
    action_required: bool
    suggested_action: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "content": self.content,
            "severity": self.severity,
            "action_required": self.action_required,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp,
        }


@dataclass
class VerificationResult:
    """Result of verifying a conclusion or correlation."""
    
    item_type: str  # insight, correlation, hypothesis
    item_id: str | int
    is_consistent: bool
    issues_found: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # Positive or negative adjustment
    verification_notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "item_type": self.item_type,
            "item_id": self.item_id,
            "is_consistent": self.is_consistent,
            "issues_found": self.issues_found,
            "confidence_adjustment": self.confidence_adjustment,
            "verification_notes": self.verification_notes,
        }


class AgentState(TypedDict):
    """
    State for the Deep Research Agent.
    
    This state flows through the LangGraph and accumulates
    intelligence findings, correlations, and analysis.
    
    Includes Multi-step Reasoning fields for deeper analysis:
    - Hypotheses tracking with evidence
    - Reasoning trace for explainability
    - Sub-task decomposition
    - Self-reflection notes
    - Verification results
    """
    
    # === Core Task ===
    task: str  # The research task/question
    context: dict[str, Any]  # Additional context provided
    
    # === Conversation ===
    messages: Annotated[list[BaseMessage], add_messages]
    
    # === Research Planning ===
    research_plan: dict[str, Any] | None  # Serialized ResearchPlan
    current_phase: str  # ResearchPhase value
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum iterations allowed
    
    # === Task Decomposition (Multi-step Reasoning) ===
    sub_tasks: list[dict[str, Any]]  # List of SubTask dicts
    current_sub_task_id: str | None  # ID of the sub-task being worked on
    task_complexity: str  # simple, moderate, complex
    
    # === Hypothesis Management (Multi-step Reasoning) ===
    hypotheses: list[dict[str, Any]]  # List of Hypothesis dicts
    active_hypothesis_id: str | None  # ID of hypothesis being tested
    
    # === Reasoning Trace (Multi-step Reasoning) ===
    reasoning_trace: list[dict[str, Any]]  # List of ReasoningStep dicts
    reasoning_depth: int  # Current depth of reasoning (increases with each step)
    chain_of_thought: list[str]  # Current chain of thought being built
    
    # === Intelligence Collection ===
    findings: list[dict[str, Any]]  # List of Finding dicts
    pending_queries: list[dict[str, Any]]  # Queries to execute
    executed_queries: list[dict[str, Any]]  # Queries already executed
    
    # === Analysis ===
    correlations: list[dict[str, Any]]  # List of Correlation dicts
    key_insights: list[str]  # Important insights extracted
    uncertainties: list[str]  # Areas of uncertainty
    
    # === Self-Reflection (Multi-step Reasoning) ===
    reflection_notes: list[dict[str, Any]]  # List of ReflectionNote dicts
    reflection_iterations: int  # How many times we've reflected
    needs_more_reflection: bool  # Whether another reflection pass is needed
    
    # === Verification (Multi-step Reasoning) ===
    verification_results: list[dict[str, Any]]  # List of VerificationResult dicts
    verified_insights: list[str]  # Insights that passed verification
    verified_correlations: list[dict[str, Any]]  # Correlations that passed verification
    
    # === Output ===
    executive_summary: str | None
    detailed_report: str | None
    recommendations: list[str]
    confidence_assessment: str | None
    
    # === Metadata ===
    started_at: str
    last_updated: str
    error: str | None


def create_initial_state(
    task: str,
    context: dict[str, Any] | None = None,
    max_iterations: int | None = None,
) -> AgentState:
    """Create initial state for a new research task."""
    now = datetime.utcnow().isoformat()
    
    return AgentState(
        # Core
        task=task,
        context=context or {},
        
        # Conversation
        messages=[],
        
        # Planning
        research_plan=None,
        current_phase=ResearchPhase.PLANNING.value,
        iteration=0,
        max_iterations=max_iterations if max_iterations is not None else settings.agent_max_iterations,
        
        # Task Decomposition (Multi-step Reasoning)
        sub_tasks=[],
        current_sub_task_id=None,
        task_complexity="moderate",  # Will be assessed during planning
        
        # Hypothesis Management (Multi-step Reasoning)
        hypotheses=[],
        active_hypothesis_id=None,
        
        # Reasoning Trace (Multi-step Reasoning)
        reasoning_trace=[],
        reasoning_depth=0,
        chain_of_thought=[],
        
        # Collection
        findings=[],
        pending_queries=[],
        executed_queries=[],
        
        # Analysis
        correlations=[],
        key_insights=[],
        uncertainties=[],
        
        # Self-Reflection (Multi-step Reasoning)
        reflection_notes=[],
        reflection_iterations=0,
        needs_more_reflection=False,
        
        # Verification (Multi-step Reasoning)
        verification_results=[],
        verified_insights=[],
        verified_correlations=[],
        
        # Output
        executive_summary=None,
        detailed_report=None,
        recommendations=[],
        confidence_assessment=None,
        
        # Metadata
        started_at=now,
        last_updated=now,
        error=None,
    )

