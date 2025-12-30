"""
Pydantic Schemas for Structured LLM Outputs.

These models define the expected output types for each research phase.
PydanticAI uses these directly via output_type parameter.
"""

from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# Research Plan Schema
# =============================================================================


class InitialQuery(BaseModel):
    """A planned query to execute."""
    tool: str = Field(description="Tool name to call")
    args: dict = Field(default_factory=dict, description="Tool arguments")


class ResearchPlanOutput(BaseModel):
    """Output from the planning phase."""
    objectives: list[str] = Field(description="Research objectives")
    regions_of_interest: list[str] = Field(description="Geographic regions to focus on")
    keywords: list[str] = Field(description="Keywords for searching")
    time_range: str = Field(default="7d", description="Time range for data collection")
    priority_sources: list[str] = Field(description="Priority data sources")
    initial_queries: list[InitialQuery] = Field(description="Initial queries to execute")


# =============================================================================
# Task Decomposition Schema
# =============================================================================


class SubTaskOutput(BaseModel):
    """A sub-task in the decomposition."""
    id: str = Field(description="Unique identifier for the sub-task")
    description: str = Field(description="What this sub-task should accomplish")
    focus_area: str = Field(default="thematic", description="Focus: geographic, temporal, or thematic")
    dependencies: list[str] = Field(default_factory=list, description="IDs of dependent sub-tasks")


class TaskDecompositionOutput(BaseModel):
    """Output from the decomposition phase."""
    task_complexity: Literal["simple", "moderate", "complex"] = Field(description="Overall complexity")
    decomposition_reasoning: str = Field(description="Reasoning for the decomposition")
    sub_tasks: list[SubTaskOutput] = Field(description="List of sub-tasks")


# =============================================================================
# Hypothesis Generation Schema
# =============================================================================


class HypothesisOutput(BaseModel):
    """A research hypothesis."""
    id: str = Field(description="Unique identifier (e.g., h1, h2)")
    statement: str = Field(description="The hypothesis statement")
    initial_confidence: float = Field(default=0.5, ge=0, le=1, description="Initial confidence 0-1")
    supporting_evidence: str = Field(default="", description="What evidence would support this hypothesis")
    refuting_evidence: str = Field(default="", description="What evidence would refute this hypothesis")


class HypothesisGenerationOutput(BaseModel):
    """Output from hypothesis generation phase."""
    hypotheses: list[HypothesisOutput] = Field(description="Generated hypotheses")


# =============================================================================
# Gather Output Schema
# =============================================================================


class GatherOutput(BaseModel):
    """Output from the gathering phase."""
    tools_called: list[str] = Field(default_factory=list, description="List of tools that were called")
    summary: str = Field(description="Summary of gathered intelligence")
    data_quality: Literal["complete", "partial", "insufficient"] = Field(default="partial")


# =============================================================================
# Analysis Schema
# =============================================================================


class AnalysisOutput(BaseModel):
    """Output from the analysis phase."""
    thinking: str = Field(default="", description="Step-by-step reasoning process before conclusions")
    key_insights: list[str] = Field(default_factory=list, description="Key insights from the analysis")
    uncertainties: list[str] = Field(default_factory=list, description="What remains uncertain")
    correlations: list[str] = Field(default_factory=list, description="Patterns or correlations found")
    confidence_assessment: str = Field(default="", description="Overall confidence in findings")


# =============================================================================
# Reflection Schema
# =============================================================================


class ReflectionNoteOutput(BaseModel):
    """A note from self-reflection."""
    category: str = Field(description="Category: gap_analysis, bias_check, alternative_explanation, or confidence_calibration")
    content: str = Field(description="The reflection note content")
    severity: str = Field(default="info", description="Severity: info, warning, or critical")


class ReflectionOutput(BaseModel):
    """Output from the reflection phase."""
    summary: str = Field(default="", description="Summary of reflection findings")
    reflection_notes: list[ReflectionNoteOutput] = Field(default_factory=list, description="Reflection notes")
    needs_more_investigation: bool = Field(default=False)
    next_steps: list[str] = Field(default_factory=list, description="Suggested next steps if more investigation needed")


# =============================================================================
# Verification Schema
# =============================================================================


class InsightVerification(BaseModel):
    """Verification result for an insight."""
    insight: str = Field(description="The insight being verified")
    verdict: str = Field(description="Verdict: pass, adjust, or fail")
    evidence: str = Field(default="", description="Evidence supporting the verdict")
    notes: str = Field(default="", description="Additional notes")


class VerificationOutput(BaseModel):
    """Output from the verification phase."""
    summary: str = Field(default="", description="Summary of verification results")
    insight_verifications: list[InsightVerification] = Field(default_factory=list)
    ready_for_synthesis: bool = Field(default=True)
    issues: list[str] = Field(default_factory=list, description="Any critical issues found")


# =============================================================================
# SITREP Output Schema (Final Report)
# =============================================================================


class SourceReliabilityEntry(BaseModel):
    """Reliability entry for a source in the intelligence matrix."""
    source_name: str
    reliability: Literal["A", "B", "C", "D", "E", "F"] = Field(default="B")
    credibility: Literal["1", "2", "3", "4", "5", "6"] = Field(default="2")
    grade: str = Field(default="", description="Combined grade (e.g., 'B-2')")


class ProbabilityAssessment(BaseModel):
    """Probability assessment for a scenario."""
    scenario: str
    probability_percent: int = Field(ge=0, le=100)
    timeframe: str = Field(default="12 months")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")


class TopicAnalysis(BaseModel):
    """Detailed analysis of a single topic."""
    topic_id: str
    title: str
    current_situation: str
    key_developments: list[str] = Field(default_factory=list)
    probability_assessments: list[ProbabilityAssessment] = Field(default_factory=list)
    evidence_citations: list[str] = Field(default_factory=list)


class SITREPSectionI(BaseModel):
    """Section I - Executive Intelligence Summary."""
    direct_response: str = Field(description="Direct answer to the user's query")
    key_highlights: list[str] = Field(description="3-5 critical bullet points")
    overall_confidence_percent: int = Field(ge=0, le=100, default=75)
    intelligence_quality: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"] = Field(default="GOOD")


class SITREPSectionII(BaseModel):
    """Section II - Detailed Analysis."""
    topics: list[TopicAnalysis] = Field(default_factory=list)
    cross_topic_connections: str = Field(default="")


class SITREPSectionIII(BaseModel):
    """Section III - Supporting Intelligence Analysis."""
    satellite_intel: str = Field(default="No satellite data collected.")
    news_intel: str = Field(default="No news data collected.")
    cyber_intel: str = Field(default="No cyber intelligence collected.")
    social_intel: str = Field(default="No social media data collected.")
    cross_source_validation: str = Field(default="")
    contradictions: list[str] = Field(default_factory=list)
    intelligence_gaps: list[str] = Field(default_factory=list)


class SITREPSectionIV(BaseModel):
    """Section IV - Actionable Intelligence & Recommendations."""
    immediate_actions: list[str] = Field(default_factory=list)
    monitoring_indicators: list[str] = Field(default_factory=list)
    follow_up_collection: list[str] = Field(default_factory=list)


class SITREPSectionV(BaseModel):
    """Section V - Intelligence Assessment Metadata."""
    source_reliability_matrix: list[SourceReliabilityEntry] = Field(default_factory=list)
    analytical_confidence: str = Field(default="")
    key_assumptions: list[str] = Field(default_factory=list)
    data_freshness: str = Field(default="")


class SITREPSectionVI(BaseModel):
    """Section VI - Forward Intelligence Requirements."""
    priority_collection: list[str] = Field(default_factory=list)
    early_warning_triggers: list[str] = Field(default_factory=list)


class SITREPOutput(BaseModel):
    """Complete SITREP (Situation Report) intelligence output."""
    classification: str = Field(default="OSINT / PUBLIC")
    query_summary: str = Field(description="Brief summary of the user's query")
    intelligence_sources_used: list[str] = Field(default_factory=list)
    
    section_i: SITREPSectionI = Field(description="Executive Intelligence Summary")
    section_ii: SITREPSectionII = Field(description="Detailed Analysis")
    section_iii: SITREPSectionIII = Field(description="Supporting Intelligence Analysis")
    section_iv: SITREPSectionIV = Field(description="Actionable Intelligence & Recommendations")
    section_v: SITREPSectionV = Field(description="Intelligence Assessment Metadata")
    section_vi: SITREPSectionVI = Field(description="Forward Intelligence Requirements")
