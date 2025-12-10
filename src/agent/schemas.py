"""
Pydantic schemas for structured LLM outputs.

These schemas are used with LangChain's `with_structured_output()` to ensure
reliable JSON responses from LLMs, even those that don't have native JSON mode.
"""

from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# Research Plan Schema
# =============================================================================


class QueryArgs(BaseModel):
    """Arguments for a tool query."""
    keywords: str | None = None
    country_code: str | None = None
    lat: float | None = None
    lon: float | None = None
    radius_km: int | None = None
    hours_back: int | None = None
    max_messages: int | None = None
    query: str | None = None
    entity_type: str | None = None
    metric: str | None = None


class InitialQuery(BaseModel):
    """A planned query to execute."""
    tool: str = Field(description="Tool name to call")
    args: dict = Field(default_factory=dict, description="Tool arguments")


class Coordinates(BaseModel):
    """Geographic coordinates with radius."""
    lat: float
    lon: float
    radius_km: int = 100


class ResearchPlanOutput(BaseModel):
    """Output schema for research planning."""
    objectives: list[str] = Field(description="Research objectives")
    regions_of_interest: list[str] = Field(description="Geographic regions to focus on")
    keywords: list[str] = Field(description="Keywords for searching")
    coordinates: list[Coordinates] = Field(default_factory=list, description="Geographic coordinates")
    time_range: str = Field(default="7d", description="Time range for data collection")
    priority_sources: list[str] = Field(description="Priority data sources")
    initial_queries: list[InitialQuery] = Field(description="Initial queries to execute")


# =============================================================================
# Task Decomposition Schema
# =============================================================================


class SubTask(BaseModel):
    """A sub-task in the decomposition."""
    id: str = Field(description="Unique identifier for the sub-task")
    description: str = Field(description="Description of what this sub-task should accomplish")
    focus_area: str = Field(default="thematic", description="Focus area: geographic, temporal, or thematic")
    dependencies: list[str] = Field(default_factory=list, description="IDs of sub-tasks this depends on")
    complexity: str = Field(default="moderate", description="Complexity level: simple, moderate, or complex")


class TaskDecompositionOutput(BaseModel):
    """Output schema for task decomposition."""
    task_complexity: Literal["simple", "moderate", "complex"] = Field(description="Overall task complexity")
    decomposition_reasoning: str = Field(description="Reasoning for the decomposition approach")
    sub_tasks: list[SubTask] = Field(description="List of sub-tasks")


# =============================================================================
# Hypothesis Generation Schema
# =============================================================================


class TestQuery(BaseModel):
    """A query to test a hypothesis."""
    tool: str = Field(description="Tool name to call")
    args: dict = Field(default_factory=dict, description="Tool arguments")
    expected_if_true: str = Field(default="", description="What we expect to find if hypothesis is true")


class Hypothesis(BaseModel):
    """A research hypothesis."""
    id: str = Field(description="Unique identifier (e.g., h1, h2)")
    statement: str = Field(description="The hypothesis statement")
    initial_confidence: float = Field(default=0.5, ge=0, le=1, description="Initial confidence 0-1")
    supporting_evidence_criteria: list[str] = Field(default_factory=list, description="What would support this")
    refuting_evidence_criteria: list[str] = Field(default_factory=list, description="What would refute this")
    test_queries: list[TestQuery] = Field(default_factory=list, description="Queries to test this hypothesis")


class HypothesisGenerationOutput(BaseModel):
    """Output schema for hypothesis generation."""
    reasoning_chain: list[str] = Field(default_factory=list, description="Step-by-step reasoning")
    hypotheses: list[Hypothesis] = Field(description="Generated hypotheses")


# =============================================================================
# Hypothesis Update Schema
# =============================================================================


class HypothesisUpdate(BaseModel):
    """Update to a hypothesis based on evidence."""
    hypothesis_id: str = Field(description="ID of hypothesis being updated")
    new_confidence: float = Field(ge=0, le=1, description="Updated confidence 0-1")
    new_status: str = Field(description="New status: proposed, investigating, supported, refuted, inconclusive")
    new_supporting_evidence: list[str] = Field(default_factory=list, description="New supporting evidence")
    new_contradicting_evidence: list[str] = Field(default_factory=list, description="New contradicting evidence")
    confidence_change_reason: str = Field(description="Reason for confidence change")


class HypothesisUpdateOutput(BaseModel):
    """Output schema for hypothesis updates."""
    reasoning_chain: list[str] = Field(default_factory=list, description="Step-by-step reasoning")
    hypothesis_updates: list[HypothesisUpdate] = Field(description="Updates to hypotheses")


# =============================================================================
# Analysis Schema
# =============================================================================


class HypothesisImplication(BaseModel):
    """How findings relate to a hypothesis."""
    hypothesis_id: str
    evidence_type: Literal["supporting", "contradicting", "neutral"]
    explanation: str


class Correlation(BaseModel):
    """A correlation between findings."""
    finding_ids: list[int] = Field(description="IDs of correlated findings")
    correlation_type: str = Field(description="Type: temporal, geospatial, causal, pattern")
    description: str = Field(description="Description of the correlation")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    implications: list[str] = Field(default_factory=list, description="Implications of this correlation")


class FollowUpQuery(BaseModel):
    """A follow-up query suggested by analysis."""
    tool: str = Field(description="Tool name to call")
    args: dict = Field(default_factory=dict, description="Tool arguments")
    reason: str = Field(description="Reason for this follow-up")


class AnalysisOutput(BaseModel):
    """Output schema for findings analysis."""
    thinking: str = Field(default="", description="Step-by-step reasoning process")
    key_insights: list[str] = Field(description="Key insights from the analysis")
    hypothesis_implications: list[HypothesisImplication] = Field(default_factory=list)
    correlations: list[Correlation] = Field(default_factory=list, description="Correlations found")
    uncertainties: list[str] = Field(default_factory=list, description="Uncertainties identified")
    follow_up_queries: list[FollowUpQuery] = Field(default_factory=list, description="Suggested follow-ups")


# =============================================================================
# Reflection Schema
# =============================================================================


class ReflectionNote(BaseModel):
    """A note from self-reflection."""
    category: Literal["gap_analysis", "bias_check", "alternative_explanation", "confidence_calibration"]
    content: str = Field(description="The reflection note content")
    severity: Literal["info", "warning", "critical"] = Field(default="info")
    action_required: bool = Field(default=False)
    suggested_action: str | None = Field(default=None)


class ReflectionOutput(BaseModel):
    """Output schema for self-reflection."""
    reflection_chain: list[str] = Field(default_factory=list, description="Reflection reasoning")
    reflection_notes: list[ReflectionNote] = Field(description="Reflection notes")
    needs_more_investigation: bool = Field(default=False)
    investigation_suggestions: list[FollowUpQuery] = Field(default_factory=list)


# =============================================================================
# Correlation Schema
# =============================================================================


class CorrelationOutput(BaseModel):
    """Output schema for correlation finding."""
    correlations: list[Correlation] = Field(description="Correlations found between sources")
    synthesis_notes: str = Field(default="", description="Overall synthesis of correlations")


# =============================================================================
# Verification Schema
# =============================================================================


class InsightVerification(BaseModel):
    """Verification result for an insight."""
    insight_index: int
    original_insight: str
    verdict: Literal["pass", "adjust", "fail"]
    evidence_check: str = Field(description="How well evidence supports this")
    issues: list[str] = Field(default_factory=list)
    revised_insight: str | None = Field(default=None)


class CorrelationVerification(BaseModel):
    """Verification result for a correlation."""
    correlation_index: int
    verdict: Literal["pass", "adjust", "fail"]
    temporal_check: str = Field(default="N/A")
    spatial_check: str = Field(default="N/A")
    issues: list[str] = Field(default_factory=list)
    suggested_confidence: Literal["high", "medium", "low"] | None = None


class OverallAssessment(BaseModel):
    """Overall verification assessment."""
    ready_for_synthesis: bool = Field(default=True)
    critical_issues: list[str] = Field(default_factory=list)


class VerificationOutput(BaseModel):
    """Output schema for verification."""
    verification_chain: list[str] = Field(default_factory=list, description="Verification reasoning")
    insight_verifications: list[InsightVerification] = Field(default_factory=list)
    correlation_verifications: list[CorrelationVerification] = Field(default_factory=list)
    overall_assessment: OverallAssessment = Field(default_factory=OverallAssessment)


# =============================================================================
# Synthesis Schema
# =============================================================================


class SynthesisOutput(BaseModel):
    """Output schema for final report synthesis."""
    executive_summary: str = Field(description="2-3 sentence summary of key findings")
    detailed_report: str = Field(description="Multi-paragraph detailed analysis in markdown")
    recommendations: list[str] = Field(description="Actionable recommendations")
    confidence_assessment: str = Field(description="Overall confidence level and explanation")
    methodology_note: str = Field(default="", description="Note on the reasoning process used")
