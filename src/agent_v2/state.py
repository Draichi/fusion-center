"""
Research Context and State for Agent v2.

Simple dataclasses to hold accumulated state across research phases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Finding:
    """A single piece of intelligence finding."""
    
    source: str
    source_type: str  # news, satellite, cyber, threat_intel, social
    timestamp: str
    content: dict[str, Any]
    relevance_score: float = 0.0
    confidence: str = "medium"
    location: dict[str, float] | None = None


@dataclass
class Hypothesis:
    """A hypothesis being tested during research."""
    
    id: str
    statement: str
    status: str = "proposed"  # proposed, investigating, supported, refuted, inconclusive
    confidence: float = 0.5
    supporting_evidence: list[int] = field(default_factory=list)
    contradicting_evidence: list[int] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SubTask:
    """A sub-task decomposed from the main research task."""
    
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed
    focus_area: str = "thematic"  # geographic, temporal, thematic


@dataclass 
class ReflectionNote:
    """A note from the agent's self-reflection process."""
    
    category: str  # bias_check, gap_analysis, alternative_explanation, confidence_calibration
    content: str
    severity: str = "info"  # info, warning, critical


@dataclass
class ResearchContext:
    """
    Context object holding all accumulated state during research.
    
    This is passed as dependencies to PydanticAI agents.
    """
    
    task: str
    max_iterations: int = 5
    
    # Planning
    research_plan: dict[str, Any] | None = None
    
    # Decomposition
    sub_tasks: list[SubTask] = field(default_factory=list)
    task_complexity: str = "moderate"
    
    # Hypotheses
    hypotheses: list[Hypothesis] = field(default_factory=list)
    
    # Intelligence Collection
    findings: list[Finding] = field(default_factory=list)
    executed_queries: list[dict[str, Any]] = field(default_factory=list)
    
    # Analysis
    key_insights: list[str] = field(default_factory=list)
    correlations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    
    # Reflection
    reflection_notes: list[ReflectionNote] = field(default_factory=list)
    
    # Verification
    verified_insights: list[str] = field(default_factory=list)
    verified_correlations: list[str] = field(default_factory=list)
    
    # Metadata
    iteration: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_finding(self, finding: Finding) -> int:
        """Add a finding and return its index."""
        self.findings.append(finding)
        return len(self.findings) - 1
    
    def get_findings_summary(self, max_items: int = 10) -> str:
        """Get a summary of findings for prompts."""
        if not self.findings:
            return "No findings collected yet."
        
        lines = []
        for i, f in enumerate(self.findings[:max_items]):
            content_preview = str(f.content)[:100]
            lines.append(f"[{i}] {f.source} ({f.source_type}): {content_preview}...")
        
        if len(self.findings) > max_items:
            lines.append(f"... and {len(self.findings) - max_items} more findings")
        
        return "\n".join(lines)
    
    def get_hypotheses_summary(self) -> str:
        """Get a summary of hypotheses for prompts."""
        if not self.hypotheses:
            return "No hypotheses generated yet."
        
        lines = []
        for h in self.hypotheses:
            lines.append(f"[{h.id}] {h.statement} (status: {h.status}, confidence: {h.confidence:.0%})")
        
        return "\n".join(lines)
